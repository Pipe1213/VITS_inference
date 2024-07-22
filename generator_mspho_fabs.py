import matplotlib.pyplot as plt
#import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import soundfile as sf

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/vctk_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/french_mspho_01/G_121000.pth", net_g, None) # replace the path with the path to the model you want to use

output_folder = './gen_audios/fr_wa_finetuned/80k/fab_female/' # replace with the path to the folder where you want to save the generated audio files
speaker_id = 1 # replace with the speaker id of the speaker you want to use

stn_tst = get_text(" li biːç ɛ l sɔlja s batẽ , tsɛkõk asyʁɔ k il ɛstɔ l py fwaʁ , kwɑ̃ il õ vøː õ vɔjɛdzøː ki s avɑ̃sɔ , abijiː avu õ paltɔ . i s õ mɛtu d akwaʁ pɔ vɛj kiː aʁivʁɔ l pʁymiː a fwɛːʁ ʁitiʁe l paltɔ ɔː vɔjɛdzøː , s  ɛ si la ki sʁɛ ʁwɛːtiː kɔm li py fwaʁ . tɔ d õ koː , li biːç s a mɛtu a ʃɔfle di tɔt sɛ fwas . m ɛ̃ py k ɛl ʃɔflɔ , py ki l vɔjɛdzøː sɛʁɔ s paltɔ tɔt ɔːtuː d ly . ɛ al fẽ , li biːç a aʁɛte di sajiː di lji fwɛːʁ ʁitiʁe s paltɔ . aloːʁ , li sɔlja a km ɛ̃siː a ʁlyːʁ , ɛ ɔː dbu d õ mum ɛ̃ , li vɔjɛdzøː , tɔ ʁʃɑ̃di , a ʁtiʁe s paltɔ . sa fwɛː ki l biːç a b ɛ̃ dvy ʁkɔny ki l sɔlja ɛstɔ l py fwaʁ dɛ døː . ", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/daverdisse.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text(" li biːç ɛ l sɔlɔ s dispytẽ , tsɛskõk asyʁɑ̃ ki s  ɛstøː ly l py fwaʁ , kwɑ̃ i vɛjiːst õ vɔjɛdzøː ki s avɑ̃siːf , ɛfyʁle ɛ s mɑ̃tja . i tumiː ɑ̃bɛdøː d akwaʁ ki l si k aʁivʁøː l pʁymiː a disvoːtjiː l vɔjɛdzøː di s mɑ̃tja sɛʁøː ʁwɛːtiː kɔm li py fwaʁ . adõ , li biːç li pʁymiːʁ atakast a ʃɔfle di tɔt si fwas , m ɛ̃ ɔː py k ɛl sɔflef , ɔː py ki l vɔjɛdzøː si ʁatwaʁtjiːf d ɛ̃ s mɑ̃tja , ɛjɛ al fẽ , ɛl ʁinõsast a l disvoːtjiː . adõ s  ɛ l sɔlɔ k atakast a lume , ɛj ɔd bu d õ mum ɛ̃ , li vɔjɛdzøː , bẽ ɛʃɑ̃di , wasta s mɑ̃tja . sa fwɛː ki l biːç diva bẽ ʁiknɔx ki s  ɛstøː l sɔlja li py fwaʁ di zɛl døː . ", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/liege.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text(" li biːç ɛ l sɔlɔ s dispitẽ , tsɛskõk asyʁɑ̃ ki s  ɛstøː ly l py fwaʁ , kwɑ̃ i vɛjiːst õ vɔjɛdzøː ki s avɑ̃siʃøː , ɛfyʁle ɛ s mɑ̃tja . i tumiː ɑ̃bɛdøː d akwaʁ ki l si k aʁivʁøː l pʁymiː a disvoːtjiː l vɔjɛdzøː di s mɑ̃tja sɛʁøː ʁwɛːtiː kɔm li py fwaʁ . adõ , li biːç li pʁymiːʁ atakast a ʃɔfle di tɔt si fwas , m ɛ̃ ɔː py k ɛl ʃɔflef , ɔː py ki l vɔjɛdzøː si ʁatwaʁtjiːf d ɛ̃ s mɑ̃tja , ɛj al fẽ , ɛl ʁinõsast a l disvoːtjiː . adõ k ɛ l sɔlɔ k atakast a lume , ɛj ɔː dbu d õ mum ɛ̃ , li vɔjɛdzøː , bẽ ɛʃɑ̃di , wasta s mɑ̃tja . sa fwɛː ki l biːç diva bẽ ʁiknɔx ki k ɛstøː l sɔlja li py fwaʁ di zɛl døː . ", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/liege2_norm.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text(" li biːç ɛ l sɔlɛː ɛst ɛ̃ ɛ maʁɡaj . ɛ tsik ɛ tsak , ɛ tsik ɛ tsak , ɔn paskɛj ki n ef põ d f ɛ̃ , dzi vɔ di . paski ʃak ɛ̃ d zɛl diʒef k il astef li py fwaʁ . m ɛ̃ õ bɛː dzuː , vɔla k i ʁɛskõtʁɛ õ vɔjadzɛyː k aʁivef pal tʁ ɛ̃ d mɔːʁloːj ɛ k a diʃ ɛ̃dy al ɡɔːʁ di ʒmɛl . mõdjy tɔdi ! kẽ apɔtikɛːʁ ! k il ef mɔː l ɛːʁ avu sɛ tsfɛ ɛmakʁalɛ ! il ɛstef dʁɔldim ɛ̃ aɡajɔlɛ , tɔ ʁavoːtjɛ d ɛ̃ õ viː paltɔ tʁawɛ pa lɛ mit . õ vʁɛː bʁibɛyː ! mɛtɑ̃ nɔ d akwaʁ ! dist i õk dɛ døː laskɔːʁ . li pʁɛmiː k aʁiːf a fwaʁsi l ɔm si mɔːʁloːj a ʁsatsɛ s paltɔ , sɛʁɛ ʁwɛːtɛ kɔm õ vʁɛː kastɔːʁ . adõ , a ! ken aʁɛts , mɛ dz ɛ̃ ! li biːç a km ɛ̃sɛ a ʃyflɛ di tɔt sɛ fwas , ɛ a stɔːʁɛ tɔ lɛ kɑ̃tja d sɛlina . maʁja dɛjiː ! ken atɛlɛːj vɛːsi ! m ɛ̃ dipy k ɛl ʃyflef , dipy ki nɔs ɔm tʁõnef ɛ s ʁakʁapɔtef d ɛ̃ s kazak . al f ɛ̃ dɛ f ɛ̃ , li biːç , tɔt disbɔːtse , a lɛjiː tyme lɛ bʁɛ . a ! s  ɛ pɔl djɔːl , dist ɛl . dzi n ɛː n ɛ̃ ply liː fe ʁoːste s paltɔ , nɔdidzɔ ! nɔs sɔlɛː ni fjef põ d b ɛ̃ . m ɛ̃ i n ɛstef n ɛ̃ bjɛs , ɛ il ɛnn a pʁɔfitɛ pɔ km ɛ̃sɛ a lyːʁ , a lyːʁ telm ɛ̃ k il a mɛty sɛ nwɔːʁɛ bɛʁik pɔ n n ɛ̃ jɛs li m ɛ̃m asbløːwi . ɔn mjɛt py tɔːʁ , l ɔm di mɔːʁloːj k ef ɔsi tsoː k d ɛ̃ õ fɔʁ , a fini pa s dismusɛ . sa fɛː  ɛ̃si ki l biːç , ɛll a stiː fwaʁse d ʁiknɔx , ki l sɔlɛː ef ɡɑ̃ɲɛ , ɛ ki s ɡʁɑ̃divɛyː la ɛstef vʁɛːm ɛ̃ l py fwaʁ . ", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/marche_en_fammenne.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text(" li biːç ɛ l sɔlja ɛstẽst ɛ maʁɡaj . ɛ tsik ɛ tsak , ɛ tsik ɛ tsak , ɛn paskɛj ki n af põ d fẽ , vɔ di dz . paski tsɛkõk di zɛl diʒøː k il ɛstøː l py fwaʁ . m ɛ̃ õ bja dzuː , vɔla k i ʁɛskõtʁɛ õ vɔjɛdzøː k aʁivef pal tʁẽ d mɔːʁloːj ɛ k a diʃ ɛ̃du al ɡɔːʁ di dzmɛl . mõdzy tɔdi ! kẽ apɔtikɔːʁ ! k il af mɔː l ɛːʁ avu sɛ tsvja ɛmakʁale ! il ɛstøː dʁɔldim ɛ̃ aɡajɔle , tɔ ʁavoːtjiː d ɛ̃ õ viː paltɔ tʁawe pa lɛ mɔt . õ vʁɛːj bʁibøː !  mɛtɑ̃ nɔ d akwaʁ !  dist i õk dɛ døː laskɔːʁ .  li pʁymiː k aʁiːf a fwaʁsi l ɔm di mɔːʁloːj a ʁsɛtsiː s paltɔ , si la sɛʁɛ ʁwɛːtiː kɔm õ vʁɛː kastɔːʁ . adõ , a ! ken aʁɛts , mɛ dz ɛ̃ ! li biːç a km ɛ̃siː a ʃɔfle di tɔt sɛ fwas , ɛ a stɔːʁe tɔ lɛ kɑ̃tja d f .  maʁja dɛjiː ! ken atɛlɛːj vɛːsi !  m ɛ̃ dipy k ɛl ʃɔflef , dipy ki nɔs ɔm tʁõnef ɛ s ʁakʁapɔte d ɛ̃ s kazak . al fẽ dɛ fẽ , li biːç , tɔt disbɔːtsɛj , a lɛjiː tume lɛ bʁɛs .  a ! k ɛ pɔl djɔːl , diʒa t ɛl . dzi n lji a savu fe ʁwaste s paltɔ , nɔdidzɔ !  nɔs sɔlja n fiʒøː põ d bẽ . m ɛ̃ i n ɛstøː nẽ bjɛs , ɛ il ɛnn a pʁɔfite pɔ km ɛ̃siː a lyːʁ , a lyːʁ telm ɛ̃ k il a mɛtu sɛ nwɛʁɛ bɛʁik pɔ n nẽ z ɛs ly m ɛ̃m asblawi . ɛn mjɛt py tɔːʁ , l ɔm di mɔːʁloːj , k af ɔsi tsoː k d ɛ̃ õ fɔʁ , a fini pa s dismusiː . sa fwɛː  ɛ̃si ki l biːç , ɛll a stiː fwaʁsɛj di ʁiknɔx , ki l sɔlja af wɑ̃ɲiː , ɛ ki s ɡʁɑ̃divøː la ɛstøː vɔːʁm ɛ̃ l py fwaʁ . ", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/marche_en_fammenne_norm.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text(" li biːç ɛ l sɔlja ɛstẽ ki s maʁɡajẽ pɔ sawɛ kiː ski , dɛ døː , ɛstøː l py fwaʁ . m ɛ̃ s koː la , la k i vɛjɛ õ tsminɔː k aʁivef pjim pjam , d ɛ̃ õ bja nuː tsoː paltɔ . ladsy , i s mɛtɛ d akwaʁ pɔ ssi , li si ki paʁvẽʁøː l pʁymiː a lji hape s mɑ̃tja , sa sʁøː ly ki sʁøː ʁwɛːtiː kɔm li py fwaʁ . adõ , la k li biːç si mɛt a ʃɔfle tɑ̃ k ɛl pu . m ɛ̃ nõ py , py sk ɛl ʃɔflef , py ski l ʁɔtøː s ʁakafyːlef d ɛ̃ s ɡʁɑ̃ paltɔ . sa fwɛː k ɛll a lɛːʃiː uf . a s mum ɛ̃ la , la ki l sɔlja s mɛt a lyːʁ kɔm kwɑ̃ i lyː dɛ kwat kɔste . ɛ s  ɛ ki , apʁɛ ɛn hapɛːj , la ki l pɔʁmõnøː a stiː ʁɛʃɑ̃di , ɛ il a tiʁe s mɑ̃tja . sa fwɛː k li biːç a bẽ dvu ʁkɔnɔx ki l sɔlja ɛstøː l py fwaʁ di zɛl døː . ", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/transinne1.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text(" s  ɛstøː l biːç ɛ l sɔlja k ɛstẽ hiɲ ɛ haɲ . paski pɔkwɛ ? paski tsɛk di zɛl fɔʁbatøː k il ɛstøː l py fwaʁ . a õ mɛtu mum ɛ̃ , la k il avizɛ õ viː poːf ki s apõtef tɔ duːsm ɛ̃ . l ɔm s avøː ʁakamizɔle d ɛ̃ õ ɡʁɑ̃ lɔːts kabɑ̃ d kyːʁ . adõ , i tyzɛ a sajiː , tsɛk a s tuː , di lji fe sɛtsiː s paltɔ , ɛ buʃiː l maʁtsiː dzy ki l si y l sɛn k adjɛʁsɛjʁøː s koː sɛʁøː ʁɛliː kɔm li py fwaʁ . s  ɛ l biːç ki km ɛ̃s , ka õ lɛː tɔdi lɛ kmɛʁ fe lɛ pʁymiːʁ . ɛ vɔ l la ki s mɛt a ʃɔfle ɛ ʁaʃɔfle , ɛ ʃɔfɛl my kɔ . m ɛ̃ bʁɔs di ɡat ! py sk ɛl ʃɔflef , py ki l viː ʁoːliː ʁihɛtsiːf si kapɔt apʁɛ ly , tɑ̃tja ki , nɔs damabõt a dvu lɛjiː uf . s  ɛstøː l tuː dɔ sɔlja . il atak a ʁlyːʁ , ɛ ʁɡlati kɔm kwɑ̃ õ kʁɛf di tsoː ɔː kwẽs d awus . sa fwɛː k nɔs pɔʁmõnøː a stiː ʁɛʃɑ̃di , si a t i hine s paltɔ dzy . ɛbẽ daboːʁ , ɛːwdiʁɔtsm ɛ̃ , s  ɛ l sɔlja k a stiː ʁiknɔʃu li py fwaʁ di zɛl døː . ", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/transinne2.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)



