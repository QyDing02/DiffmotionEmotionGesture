import logging
import sys
import os
import numpy as np
import pickle
from src.build_vocab import Vocab
from torch.nn.modules import loss
from torch.nn.utils import weight_norm
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated

from src.models.components.TimeGradTrainingNetwork.utils import weighted_average
from src.models.components.TimeGradTrainingNetwork.modules import GaussianDiffusion, DiffusionOutput, MeanScaler, \
    NOPScaler
from src.models.components.TimeGradTrainingNetwork.modules.distribution_output import GaussianDiag

from .epsilon_theta import EpsilonTheta
from src import utils
from tqdm import tqdm
import ipdb
from src.models.components.TimeGradTrainingNetwork.modules.act_norm import ActNorm2d
from src.models.components.TimeGradTrainingNetwork.modules.permute2d import Permute2d


log = utils.get_pylogger(__name__)


# import logging as log
#
# logger = log.getLogger('test')
# logger.setLevel(level=logging.INFO)
# fh = logging.FileHandler('/home/zf223669/Mount/pytorch-ts-2/test.log', mode='w')
# ch = logging.StreamHandler()
# ch.setLevel(level=logging.INFO)
# logger.addHandler(fh)
# logger.addHandler(ch)

class LinearZeroInit(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        # init
        # print("-----------------LinearZeroInit----------------")
        self.weight.data.zero_()
        self.bias.data.zero_()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, freeze_wordembed, hidden_size, word_f, n_layer, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()
        # config
        self.freeze_wordembed = freeze_wordembed
        self.hidden_size = hidden_size
        self.word_f = word_f
        self.n_layer = n_layer
        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            #print(pre_trained_embedding.shape)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=self.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [self.hidden_size] * self.n_layer
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], self.word_f)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


class BasicBlock(nn.Module):
    """ based on timm: https://github.com/rwightman/pytorch-image-models """
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU,   norm_layer=nn.BatchNorm1d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,  stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes), 
            )
        else: self.downsample=None
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x

class WavEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(1, 32, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(32, 32, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(32, 32, 15, 1, first_dilation=7, ),
                BasicBlock(32, 64, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(64, 64, 15, 1, first_dilation=7),
                BasicBlock(64, self.out_dim, 15, 6,  first_dilation=0,downsample=True),
            )
        
    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1) 
        out = self.feat_extractor(wav_data) 
        return out.transpose(1, 2) 
    

class TimeGradTrainingNetwork(nn.Module):
    @validated()
    def __init__(
            self,
            input_size: int,  # 972
            num_layers: int,  # 2
            num_cells: int,  # 512
            cell_type: str,  # LSTM / GRU
            prediction_length: int,  # 24
            dropout_rate: float,
            target_dim: int,  # 370
            conditioning_length: int,  # 100
            diff_steps: int,
            loss_type: str,
            beta_end: float,
            beta_schedule: str,
            residual_layers: int,
            residual_channels: int,
            dilation_cycle_length: int,
            scaling: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        log.info(f"-------------------Init TimeGradTrainingNetwork----------------")
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.scaling = scaling
        self.num_cells = num_cells
        self.dropout_prob = 0.3

        self.cell_type = cell_type
        self.audio_f = 128 #128
        self.emotion_f = 8
        self.emotion_dims = 8
        self.emotion_embedding = True
        self.pose_dim = 141
        self.pre_frames = 4
        self.word_f = 0
        self.word_index_num = 5793
        self.word_dims = 300
        # self.init_rnn = True
        
        # data and path
        self.mean_pose = np.load("/home/lingling/code/DiffmotionEmotionGesture_v1/data/beat_cache/beat_4english_15_141_v3/train/bvh_rot/bvh_mean.npy")
        self.std_pose = np.load("/home/lingling/code/DiffmotionEmotionGesture_v1/data/beat_cache/beat_4english_15_141_v3/train/bvh_rot/bvh_std.npy")
        self.vocab_path = '/home/lingling/code/DiffmotionEmotionGesture_v1/data/beat_cache/beat_4english_15_141_v3/vocab.pkl'

        #audio 
        self.audio_encoder = WavEncoder(self.audio_f)
        self.in_size = self.audio_f  + self.pose_dim + 1
        self.hidden_size = 256 #512
        
        self.n_layer = 4 #4

        
        # emotion
        self.emotion_embedding = None
        if self.emotion_f != 0:
            self.in_size += self.emotion_f
            
            self.emotion_embedding = nn.Sequential(
                nn.Embedding(self.emotion_dims, self.emotion_f),
                nn.Linear(self.emotion_f, self.emotion_f) 
            )

            self.emotion_embedding_tail = nn.Sequential( 
                nn.Conv1d(self.emotion_f, 8, 9, 1, 4),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(8, 16, 9, 1, 4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(16, 16, 9, 1, 4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(16, self.emotion_f, 9, 1, 4),
                nn.BatchNorm1d(self.emotion_f),
                nn.LeakyReLU(0.3, inplace=True),
            )
    
        # text
        self.text_encoder = None   
        if self.word_f is not 0:
            self.in_size += self.word_f
            # self.lang_model = torch.load(self.vocab_path)  #deepcopy TODO
            self.text_encoder = TextEncoderTCN(False, self.hidden_size, self.word_f, self.n_layer, self.word_index_num, self.word_dims, pre_trained_embedding=None,
                                           dropout=self.dropout_prob)
        
        self.LSTM = nn.LSTM(self.in_size, hidden_size=self.hidden_size, num_layers=self.n_layer, batch_first=True,
                        bidirectional=True, dropout=self.dropout_prob)
            
        # self.full = nn.Linear(36266,34 * num_cells)
        self.full = nn.Linear(input_size,34 * num_cells)

        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.denoise_fn = EpsilonTheta(  # εΘ learn this function
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )

        self.diffusion = GaussianDiffusion(  # Most Importance
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        )
        self.proj_dist_args = self.distr_output.get_args_proj(num_cells)
        self.normal_distribution = GaussianDiag()

        # self.proj_dist_args = self.distr_output.get_args_proj(num_cells)

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )
        # self.BatchNorm = nn.BatchNorm1d(num_features=45)
        if self.scaling:
            # self.actnorm = ActNorm2d(45, 1.0)
            self.actnorm = ActNorm2d(141, 1.0)
        # self.shuffle = Permute2d(45, shuffle=True)

        # self.linear = LinearZeroInit(num_cells, 512)

        self.forwardCount = 1
        # if self.scaling:
        #     self.scaler = MeanScaler(keepdim=True)
        # else:
        #     self.scaler = NOPScaler(keepdim=True)

    def distr_args(self, rnn_outputs: torch.Tensor):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        (distr_args,) = self.proj_dist_args(rnn_outputs)

        # # compute likelihood of target given the predicted parameters
        # distr = self.distr_output.distribution(distr_args, scale=scale)

        # return distr, distr_args
        return distr_args

    def forward(
            self,
            trainer,
            x_input: torch.Tensor,
            cond: torch.Tensor, 
            word: torch.Tensor, 
            id: torch.Tensor, 
            emo: torch.Tensor):
        # x_input_scaled, _ = self.actnorm(x_input, None, reverse=False)
        # combined_input = torch.cat((x_input_scaled, cond), dim=-1)
        #
        # rnn_outputs, state = self.rnn(combined_input)
        # distr_args = self.distr_args(rnn_outputs=rnn_outputs)
        # likelihoods = self.diffusion.log_prob(x_input_scaled, distr_args).unsqueeze(-1)
        # log.info(f'likelihoods shape: {likelihoods}')

        # without x_input to rnn

        # index_embeddings = self.embed(self.time_index)
        # combined_cond = torch.cat((cond, index_embeddings), dim=-1)
        datamodule = trainer.datamodule
        # combined_cond = cond
        # if datamodule.input_size == 792:
        # if self.scaling:
        #     x_input, _ = self.actnorm(x_input, None, reverse=False)
        # combined_cond = torch.cat((x_input, cond), dim=-1)
        # target_dimension_indicator = torch.arange(0, 45).repeat((80,1)).reshape((80,45)).cuda()
        # index_embeddings = self.embed(target_dimension_indicator)
        # repeated_index_embeddings = (
        #     index_embeddings.unsqueeze(1).expand(-1,95,-1,-1).reshape((-1,95,45))
        # )
        # combined_cond = torch.cat((combined_cond, repeated_index_embeddings), dim=-1)

        ## src Diffmotion
        # rnn_outputs, _ = self.rnn(combined_cond)  # rnn的hiddensize就是outputs——size

        ## 用一层全连接层
        # rnn_outputs = self.full(combined_cond)
        # # print("full connection ____________")
        # # print(combined_cond.shape, rnn_outputs.shape)
        # rnn_outputs = rnn_outputs.reshape(256, 34, self.num_cells)
        # # print("rnn size: ", rnn_outputs.shape)
        
        pre_pose = x_input.new_zeros((x_input.shape[0], x_input.shape[1], x_input.shape[2] + 1)).cuda()
        pre_pose[:, 0:self.pre_frames, :-1] = x_input[:, 0:self.pre_frames]
        pre_pose[:, 0:self.pre_frames, -1] = 1 
        
        text_feat_seq = audio_feat_seq = emo_feat_seq =  None
        in_data = None
        
        if self.emotion_embedding:
            emo_feat_seq = self.emotion_embedding(emo)
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq) 
            emo_feat_seq = emo_feat_seq.permute([0,2,1])

        # if word is not None:
        #     text_feat_seq, _ = self.text_encoder(word)
            
        if cond is not None:
            audio_feat_seq = self.audio_encoder(cond) 
            
        if  audio_feat_seq.shape[1] != x_input.shape[1]:
            diff_length = x_input.shape[1] - audio_feat_seq.shape[1]
            audio_feat_seq = torch.cat((audio_feat_seq, audio_feat_seq[:,-diff_length:, :].reshape(1,diff_length,-1)),1)
            
        in_data = torch.cat((in_data, audio_feat_seq), 2)if in_data is not None else audio_feat_seq
        
        # if text_feat_seq is not None:
        #     in_data = torch.cat((in_data, text_feat_seq), dim=2)
            
        if emo_feat_seq is not None:
            in_data = torch.cat((in_data, emo_feat_seq), dim=2)
        
        in_data = torch.cat((pre_pose,in_data), 2)
       
        rnn_outputs, _ = self.LSTM(in_data)
        # print("audio_feat_seq", audio_feat_seq.shape)
        # print("rnn_outputs", rnn_outputs.shape)
        # rnn_outputs = torch.as_tenso(rnn_outputs, dtype=None, device=None)

        distr_args = self.distr_args(rnn_outputs=rnn_outputs)
        if self.scaling:
            x_input, _ = self.actnorm(x_input, None, reverse=False) # TODO: may be problem
        # x_input_scaled = self.shuffle(x_input_scaled, False)

        # print("forward: _________________________________-")
        # print("target pose x_input size: ", x_input.shape)
        # print("combined_cond size: ", combined_cond.shape)

        likelihoods = self.diffusion.log_prob(x_input, rnn_outputs).unsqueeze(-1)

        return likelihoods, likelihoods.mean()


class TimeGradPredictionNetwork(TimeGradTrainingNetwork):
    def __init__(self, num_parallel_samples: int, bvh_save_path: str, quantile: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.init_rnn = True
        self.state = None

        self.bvh_save_path = bvh_save_path
        self.bvh_temp_path = os.path.dirname(bvh_save_path)+'/temp'
        self.num_parallel_samples = num_parallel_samples #200
        self.quantile = quantile
        log.info(f"-------------------Init TimeGradPredictionNetwork----------------")
        self.inited_rnn = False
        self.test_demo = "/home/lingling/code/DiffmotionEmotionGesture_v1/data/beat_cache/beat_4english_15_141_v3/test/bvh_rot_vis/"
        self.pose_length = 34
        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        # self.shifted_lags = [l - 1 for l in self.lags_seq]

    # def prepare_cond(self, jt_data, ctrl_data):
    #     # log.info(f'type of jt_data {jt_data.device} ,ctrl_data : {ctrl_data.device}')
    #     jt_data = jt_data.cuda()
    #     ctrl_data = ctrl_data.cuda()
    #     # log.info(f'to cuda type of jt_data {jt_data.device} ,ctrl_data : {ctrl_data.device}')
    #     nn, seqlen, n_feats = jt_data.shape
    #     # log.info('prepare_cond........')
    #     jt_data = torch.reshape(jt_data, (nn, seqlen * n_feats))  # jt_data [80,225]
    #     nn, seqlen, n_feats = ctrl_data.shape
    #     ctrl_data = torch.reshape(ctrl_data, (nn, seqlen * n_feats))  # ctrl_data [80,702]
    #     # log.info(f'jt_data shape: {jt_data.shape}, ctrl_data shape: {ctrl_data.shape}')
    #     # #jt_data [80,225]  ctrl_data [80,702]
    #     cond = torch.cat((jt_data, ctrl_data), 1)  # [80,927]
    #     cond = torch.unsqueeze(cond, 1)  # [80,1,927]

    #     return cond

    # def sampling_decoder(
    #         self,
    #         autoreg: torch.Tensor,
    #         begin_states: Union[List[torch.Tensor], torch.Tensor],
    #         control_all: torch.Tensor,
    #         sampled_all: torch.Tensor,
    #         seqlen: int,
    #         n_lookahead: int,
    #         # scale: torch.Tensor,
    # ) -> torch.Tensor:
    #     # torch.set_printoptions(threshold=sys.maxsize)
    #     np.set_printoptions(threshold=500)
    #     future_samples = sampled_all.cpu().numpy().copy()  # [0,0,0,0,0,,,,,,] shape:[80,380,45]
    #     # log.info(f'future_samples :{future_samples.shape}')
    #     # if self.scaling:
    #     #     self.diffusion.scale = scale
    #     autoreg = autoreg
    #     states = begin_states
    #     self.prediction_length = future_samples.shape[1]
    #     # self.actnorm.inited = True
    #
    #     # for each future time-units we draw new samples for this time-unit
    #     # and update the state
    #
    #     for k in tqdm(range(control_all.shape[1] - seqlen - n_lookahead)):
    #         # log.info(f'prediction_length = {self.prediction_length}')
    #         # log.info(f'k = {k} ; control_length  = {k + seqlen + 1 + n_lookahead} ')
    #         control = control_all[:, k:(k + seqlen + 1 + n_lookahead), :]
    #         # log.info(f'control shape: {control.shape}')
    #         # log.info(f'autoreg shape: {autoreg.shape}')
    #         combined_cond = self.prepare_cond(autoreg, control)
    #         # log.info(f'cond shape: {combined_cond.shape}')
    #         # z = self.normal_distribution.sample([80, 1, 45], 1e-08, "cuda:0")
    #         # log.info(f'z shape: {z.shape}')
    #
    #         rnn_outputs, state = self.rnn(combined_cond)
    #         # distr_args = self.distr_args(rnn_outputs=rnn_outputs)
    #         new_samples = self.diffusion.sample(cond=rnn_outputs)
    #         new_samples, _ = self.actnorm(new_samples, None, reverse=True)
    #         # log.info(f'new_samples : {type(new_samples)}, device: {new_samples.device}')
    #         new_samples = new_samples.cpu().numpy()[:, 0, :]
    #         # new_samples = new_samples[:, 0, :]
    #         future_samples[:, (k + seqlen), :] = new_samples
    #         # (batch_size, seq_len, target_dim)
    #         # future_samples.append(new_samples)
    #
    #         # log.info(f'new_samples: \n{new_samples} \n future_samples: {future_samples}')
    #         autoreg = autoreg.cpu().numpy()
    #         # log.info(
    #         #     f'new_samples shape: {new_samples.shape} , future_samples shape: {future_samples.shape}, autoreg shape: {autoreg.shape}')
    #         # log.info(f'new_samples[:, None, :] shape : {new_samples[:, None, :].shape}')
    #         autoreg = np.concatenate((autoreg[:, 1:, :].copy(), new_samples[:, None, :]), axis=1)
    #         autoreg = torch.from_numpy(autoreg).cuda()
    #         # print(f'--->autoreg shape:{autoreg.shape} \n {autoreg}')
    #     # (batch_size * num_samples, prediction_length, target_dim)
    #     # samples = torch.cat(future_samples, dim=1)
    #     log.info(f'samples length: {future_samples.size} type {type(future_samples)} future_samples: {future_samples} ')
    #     # (batch_size, num_samples, prediction_length, target_dim)
    #     return future_samples
    def forward(self,x, cond, word, id, emo,batch_idx, trainer) -> torch.Tensor:
        log.info("Prediction forward .....................................")
        datamodule = trainer.datamodule
        num_batch = datamodule.batch_size
        batch, n_timesteps, n_feats = x.shape
        sampled_all = torch.zeros((batch, n_timesteps, n_feats))
        log.info('Sampling_decoder')
        

        in_data = None
        text_feat_seq = audio_feat_seq = emo_feat_seq =  None
        pre_pose = x.new_zeros((x.shape[0], x.shape[1], x.shape[2] + 1)).cuda()
        pre_pose[:, 0:self.pre_frames, :-1] = x[:, 0:self.pre_frames]
        pre_pose[:, 0:self.pre_frames, -1] = 1
        # beat
        # if word is not None:
        #     text_feat_seq, _ = self.text_encoder(word)
        
        if cond is not None:
            audio_feat_seq = self.audio_encoder(cond) 
            
        if self.emotion_embedding:
            emo_feat_seq = self.emotion_embedding(emo)
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq) 
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            
        if  audio_feat_seq.shape[1] != n_timesteps:
            diff_length = n_timesteps- audio_feat_seq.shape[1]
            audio_feat_seq = torch.cat((audio_feat_seq, audio_feat_seq[:,-diff_length:, :].reshape(1,diff_length,-1)),1)
        in_data = torch.cat((in_data, audio_feat_seq), 2)if in_data is not None else audio_feat_seq
        
        # if text_feat_seq is not None:
        #     in_data = torch.cat((in_data, text_feat_seq), dim=2)
            
        if emo_feat_seq is not None:
            in_data = torch.cat((in_data, emo_feat_seq), dim=2)
            
        in_data = torch.cat((pre_pose,in_data), 2)
        
        rnn_outputs, self.state = self.LSTM(in_data)
        img = self.normal_distribution.sample((batch * 1, n_timesteps, self.pose_dim), 1,
                                                  device=cond.device)
        new_samples = self.diffusion.sample(cond=rnn_outputs, img=img)
        if self.scaling:
            new_samples, _ = self.actnorm(new_samples, None, reverse=True)
        future_samples = new_samples.reshape(1,n_timesteps,141)

        future_samples = future_samples.cpu().numpy().copy()
        # seqlen = datamodule.seqlen
        # n_lookahead = datamodule.n_lookahead

        # batch, n_timesteps, n_feats = autoreg_all.shape  #autoreg_all.shape: torch.Size([1, 855, 141])
        # log.info(f"autoreg_all.shape: {autoreg_all.shape}") 
        # sampled_all = torch.zeros((batch, n_timesteps - n_lookahead, n_feats))  # sampled_all.shape: torch.Size([1, 835, 141])
        # log.info(f"sampled_all.shape: {sampled_all.shape}")
        # autoreg = torch.zeros((batch, seqlen, n_feats), dtype=torch.float32)  # autoreg.shape: torch.Size([1, 5, 141])
        # log.info(f"autoreg.shape: {autoreg.shape}")

        # sampled_all[:, :seqlen, :] = autoreg  # start pose [0,0,0,0,0] sampled_all.shape: torch.Size([1, 835, 141])
        # log.info(f"sampled_all.shape: {sampled_all.shape}") #

  
        # np.set_printoptions(threshold=500)
        # future_samples = sampled_all.cpu().numpy().copy()  # [0,0,0,0,0,,,,,,] shape:[80,380,45]
        # future_samples = sampled_all

        # for each future time-units we draw new samples for this time-unit
        # and update the state
        
        # def repeat(tensor, dim=0):
        #     return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # repeated_states = None
        
        # log.info(f"control_all.shape: {control_all.shape}")
        # control = control_all[:, 0:(seqlen + 1 + n_lookahead), :]
        # log.info(f"control.shape: {control.shape} ")
        
        # combined_cond = self.prepare_cond(autoreg, control) ##删掉
 
        # img = self.normal_distribution.sample((datamodule.batch_size, 1, 45), 1,
        #                                       device=combined_cond.device)
        # # if self.scaling:
        # #     img, _ = self.actnorm(img, None, reverse=True)
        # combined_cond = torch.cat((img, combined_cond), dim=-1)

        
        # diffmotion: rnn_outputs, self.state = self.rnn(combined_cond)
        
        # if self.cell_type == "LSTM":
        #     repeated_states = [repeat(s, dim=1) for s in self.state]
        # else:
        #     repeated_states = repeat(self.state, dim=1)
        # self.inited_rnn = True

        # repeated_control_all = repeat(control_all)
        # repeated_autoreg = repeat(autoreg)
        # repeated_future_samples = repeat(future_samples)

        # for k in tqdm(range(control_all.shape[1] - seqlen - n_lookahead - 1)):
        #     # ipdb.set_trace(context=5)
        #     repeated_control = repeated_control_all[:, (k + 1):((k + 1) + seqlen + 1 + n_lookahead), :]
        #     repeated_autoreg = repeat(autoreg)
        #     # combined_cond = self.prepare_cond(repeated_autoreg, repeated_control)
        #     combined_cond = repeated_control
        #     img = self.normal_distribution.sample((datamodule.batch_size * self.num_parallel_samples, 1, 45), 1,
        #                                           device=combined_cond.device)
            # if self.scaling:
            #     new_samples, _ = self.actnorm(img, None, reverse=True)
            # combined_cond = torch.cat((img, combined_cond), dim=-1)

            # control = control_all[:, k:(k + seqlen + 1 + n_lookahead), :]
            # combined_cond = self.prepare_cond(autoreg, control)
            # # if datamodule.input_size == 792:
            # img = self.normal_distribution.sample((80, 1, 45), 1, device=combined_cond.device)
            # combined_cond = torch.cat((img, combined_cond), dim=-1)
            # else:
            #     img = None
            # target_dimension_indicator = torch.arange(0, 45).repeat((80, 1)).reshape((80, 45)).cuda()
            # index_embeddings = self.embed(target_dimension_indicator)
            # repeated_index_embeddings = (
            #     index_embeddings.unsqueeze(1).expand(-1, 1, -1, -1).reshape((-1, 1, 45))
            # )
            # combined_cond = torch.cat((combined_cond, repeated_index_embeddings), dim=-1)
            # if not self.inited_rnn:
            #     # log.info(f"Not inited_rnn!!")
            #     # rnn_outputs, self.state = self.rnn(combined_cond)
            #     if self.cell_type == "LSTM":
            #         repeated_states = [repeat(s, dim=1) for s in self.state]
            #     else:
            #         repeated_states = repeat(self.state, dim=1)
            #     self.inited_rnn = True
            # else:
            #     # log.info(f"inited_rnn!!")
            #     # rnn_outputs, self.state = self.rnn(combined_cond, self.state)
            #     rnn_outputs, repeated_states = self.rnn(combined_cond, repeated_states)
            # distr_args = self.distr_args(rnn_outputs=rnn_outputs)

            # Diffmotion rnn_outputs, repeated_states = self.rnn(combined_cond, repeated_states)
            
            # audio_feat_seq = self.audio_encoder(combined_cond) 
            # audio_feat_seq = combined_cond
            # in_data = audio_feat_seq
            # if self.emotion_embedding:
            #     emo_feat_seq = self.emotion_embedding(emo)
            #     emo_feat_seq = emo_feat_seq.permute([0,2,1])
            #     emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq) 
            #     emo_feat_seq = emo_feat_seq.permute([0,2,1])
            #     in_data = torch.cat((in_data, emo_feat_seq), 2) if in_data is not None else emo_feat_seq
            
            # rnn_outputs, repeated_states = self.LSTM(in_data)

            # new_samples = self.diffusion.sample(cond=rnn_outputs, img=img)

            # new_samples = self.shuffle(new_samples, True)
            # new_samples = new_samples.cpu().numpy()[:, 0, :]
            # ipdb.set_trace(context=5)
            # ipdb.set_trace(context=6)
            # new_samples = new_samples[:, 0, :]
            # new_samples = new_samples.reshape(-1, self.num_parallel_samples, n_feats)
            # if self.scaling:
            #     new_samples, _ = self.actnorm(new_samples, None, reverse=True)
            # quantile_new_samples = torch.quantile(new_samples, self.quantile, dim=1)

            # if self.scaling:
            #     quantile_new_samples, _ = self.actnorm(quantile_new_samples, None, reverse=True)
            #     quantile_new_samples = torch.squeeze(quantile_new_samples, dim=1)
            # future_samples[:, (k + seqlen), :] = quantile_new_samples
            # ipdb.set_trace(context=5)
            # repeated_future_samples[:, (k + seqlen), :] = new_samples
            # future_samples = repeated_future_samples.reshape(-1, self.num_parallel_samples, n_timesteps - n_lookahead,
            #                                                  n_feats)
            # ipdb.set_trace(context=5)
            # future_samples[:, (k + seqlen), :] = new_samples
            # autoreg = autoreg.cpu().numpy()
            # autoreg = np.concatenate((autoreg[:, 1:, :].copy(), quantile_new_samples[:, None, :].cpu().numpy()), axis=1)
            # autoreg = torch.from_numpy(autoreg).cuda()

        # sampled_all_done = self.sampling_decoder(autoreg=autoreg, control_all=control_all,
        # begin_states=None, sampled_all=sampled_all, seqlen=seqlen, n_lookahead=n_lookahead, ) log.info(f'final x
        # shape: {x.shape} x = \n {x}')
        future_samples = new_samples.reshape(1,n_timesteps,self.pose_dim)

        future_samples = future_samples.cpu().numpy().copy()
        out_final = future_samples.reshape(n_timesteps,self.pose_dim)
        out_final = (out_final * self.std_pose) + self.mean_pose
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        test_seq_list = [file for file in test_seq_list if file.endswith('.bvh')]
        if not os.path.exists(f"{self.bvh_save_path}/"): 
            os.mkdir(f"{self.bvh_save_path}/")
        if not os.path.exists(f"{self.bvh_temp_path}/"): 
            os.mkdir(f"{self.bvh_temp_path}/")
        with open(f"{self.bvh_temp_path}/result_raw_{test_seq_list[batch_idx]}", 'w+') as f_real:
            for line_id in range(out_final.shape[0]): #,args.pre_frames, args.pose_length
                line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                f_real.write(line_data[1:-2]+'\n')  
        # log.info(f'sampled_all: {future_samples}, {type(future_samples)}')
        # datamodule.save_animation(future_samples, self.bvh_save_path)
        datamodule.result2target_vis(res_bvhlist = self.bvh_temp_path, save_path = self.bvh_save_path, demo_name = self.test_demo, verbose = False)
        return sampled_all
