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

_ = utils.load_checkpoint("./logs/french_ms_01/G_191000.pth", net_g, None) # replace with checkpoint path

output_folder = './gen_audios/fr_wa_ms_400K/fabs_male/' # replace with output folder
speaker_id = 0 # replace with speaker id

stn_tst = get_text("Li bijhe et l’ solea estént ki s’ margayént po sawè kî çki, des deus, esteut l’ pus foirt. Mins ç’ côp la, la k’ i veyèt on tchminåd k' arivéve pyim piam, dins on bea noû tchôd paltot. Ladsu, i s' metèt d' acoird po çci: li ci ki parvénreut l' prumî a lyi haper s' mantea, ça sreut lu ki sreut rwaitî come li pus foirt. Adon, la k' li bijhe si mete a shofler tant k' ele pout. Mins non pus, pus çk' ele shofléve, pus çki l' roteu s' racafûléve dins s' grand paltot. Ça fwait k' elle a laixhî ouve. A ç' moumint la, la ki l' solea s' mete a lure come cwand i lût des cwate costés. Et c' est ki: après ene hapêye, la ki l' pormoenneu a stî reschandi, et il a tiré s' mantea. Ça fwait k' li bijhe a bén dvou rconoxhe ki l' solea esteut l' pus foirt di zels deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/transinne1.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe et l’ solo s’ dispitént, tchaesconk assurant ki c’ esteut lu l’ pus foirt, cwand i veyît-st on voyaedjeu ki s’ avancixheut, efurlé e s’ mantea. I toumît ambedeus d’ acoird ki l’ ci k’ arivreut l’ prumî a disvôtyî l’ voyaedjeu di s’ mantea sereut rwaitî come li pus foirt. Adon, li bijhe li prumire ataca-st a shofler di tote si foice, mins å pus k’ ele shofléve, å pus ki l’ voyaedjeu si ratoirtyive dins s’ mantea ; ey al fén, ele rinonça-st a l’ disvôtyî. Adon c’ est l’ solo k’ ataca-st a loumer, ey å dbout d’ on moumint, li voyaedjeu, bén eschandi, oista s’ mantea. Ça fwait ki l’ bijhe diva bén ricnoxhe ki c’ esteut l’ solea li pus foirt di zels deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/liege2_norm.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe et l’ solea estént-st e margaye. Et tchik et tchak, et tchik et tchak : ene paskeye ki n’ av pont d’ fén, vos di dj’. Paski tchaeconk di zels dijheut k’ il esteut l’ pus foirt. Mins on bea djoû, vola k’ i rescontrèt on voyaedjeu k’ arivéve pal trén d’ Mårloye et k’ a dischindou al gåre di Djmele. Mondju todi ! Kén apoticåre ! K’ il av må l’ air avou ses tchveas emacralés ! Il esteut droldimint agayolé, tot ravôtyî dins on vî paltot trawé pa les motes. On vraiy bribeu ! « Metans nos d’ acoird ! » dit-st i onk des deus lascår. « Li prumî k’ arive a foirci l’ ome di Mårloye a rsaetchî s’ paltot, ci-la serè rwaitî come on vrai castår. Adon, a ! kéne araedje, mes djins ! Li bijhe a cmincî a shofler di totes ses foices, et a stårer tos les canteas d’ v. « Maria deyî ! Kéne atelêye vaici ! » Mins dipus k’ ele shofléve, dipus ki nost ome tronnéve et s’ racrapoter dins s’ cazake. Al fén des féns, li bijhe, tote disbåtcheye, a leyî toumer les bresses. « A ! c’ est pol diåle, dijha-t ele. Dji n’ lyi a savou fé roister s’ paltot, nodidjo ! » Nosse solea n’ fijheut pont d’ bén. Mins i n’ esteut nén biesse, et il end a profité po cmincî a lure, a lure télmint k’ il a metou ses noerès berikes po n’ nén-z esse lu-minme asblawi. Ene miete pus tård, l’ ome di Mårloye, k’ av ossi tchôd k’ dins on for, a fini pa s’ dismoussî. Ça fwait insi ki l’ bijhe, elle a stî foirceye di ricnoxhe, ki l’ solea av wangnî, et ki ç’ grandiveus la esteut vårmint l’ pus foirt.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/marche_en_fammenne_norm.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe eyet l’ solea s’ bretént, tchaeconk acertinant k’ il esteut l’ pus foirt, cwand il ont veyou on voyaedjeu avanci, ewalpé dins s’ paltot. I s’ ont metou d’ acoird kel ci k’ arivreut l’ prumî a fé oister s’ paltot å voyaedjeu, ci-la sreut rmetou come li pus foirt. Adon l’ bijhe s’ a metou a shofler di tote ses foices mins dpus k’ ele shofléve, dipus kel voyaedjeu seréve si paltot åtoû d’ lu et, al difén, l’ bijhe a rnoncî a l’ lyi fé oister. Ça fwait kel solea a-st ataké a rlure et après on moumint, el voyaedjeu, restchåfé, a oisté s’ paltot. C’ est insi kel bijhe a dvou rconoxhe kel solea esteut l’ pus foirt di zels deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/mont_saint_guibert_norm.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe et l' solea s' batént ; tchaeconk assurot k' il estot l' pus foirt, cwand il ont veu on voyaedjeu ki s' avançot, abiyî avou on paltot. I s' ont metou d' acoird po vey kî arivrot l' prumî a fwaire ritirer l' paltot å voyaedjeu : c' est ci-la ki srè rwaitî come li pus foirt. Tot d' on côp, li bijhe s' a metou a shofler di totes ses foices. Mins pus k' ele shoflot, pus ki l' voyaedjeu serot s' paltot tot åtoû d' lu. Et al fén, li bijhe a areté di sayî di lyi fwaire ritirer s' paltot. Alôr, li solea a cmincî a rlure, et å dbout d' on moumint, li voyaedjeu, tot rschandi, a rtiré s' paltot. Ça fwait ki l' bijhe a bin dvu rconu ki l' solea estot l' pus foirt des deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/daverdisse.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe et l' solea s' dispitént, tchaeconk assurant k' il esteut l' pus foirt. Tot d' on côp, il ont veyou on voyaedjeu ki s'avancixheut, moussî dins s' camizole. Il ont tcheyou d'acoird ki l' ci k' arivreut a fé tirer s' camizole å voyaedjeû, sreut rwaitî come li pus foirt. Adon l' bijhe s' a metou a shofler d' totes ses foices. Pus çk' ele shofléve, pus çki l' voyaedjeu si racrapotéve dins s' camizole. Li bijhe a baxhî les bresses. Adon, l' solea a cmincî a lure. Li voyaedjeu esteut rschandi, et il a tapé s' camizole evoye. Li bijhe, batowe, a dvou admete ki l' solea esteut bén pus foirt.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/houyet.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Nosse solea et li rosse di bijhe n' estént nén d' acoird po sawè li ké des deus esteut l' pus foirt, dijhans putoit li pus sovrin (sins pont d' chichis ni brut). Vola k' i veynut so l' pî-sinte on pormoenneu rewalpé dins s' paltot. I fwaiynut martchî ki l' ci ki lyi frè roister s' pardessu serè veyou come li pus foirt. Li bijhe si mete a shofler di totes ses foices mins å dpus ele shofele å dpus l' ome si ressere dins s' mantea (il a minme rabatou l' capuchon di s' paltot so s' tiesse...). Come ele ni parvént nén a fé disboter l' ome, li bijhe si djoke di shofler, discoraedjeye. Asteure ki c' est a lu, li solea, on ptit sorire a l' coine di ses lepes, i s' mete a rlure, a rlure... et ça tchåfe... et l' ome rissaetche si capuchon, po cminçî ; et tot d' shûte après, i rsaetche si mantea ! Li bijhe li sait bén asteure : li doûceur, çoula va bråmint mia ki l' foice.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/jemeppe_sur_sambre.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe et l' solo s' disputént, tchaesconk assurant ki c' esteut lu l' pus foirt, cwand i veyît-st on voyaedjeu ki s' avancive, efurlé e s' mantea. I toumît ambedeus d' acoird ki l' ci k' arivreut l' prumî a disvôtyî l' voyaedjeu di s' mantea sereut rwaitî come li pus foirt. Adon, li bijhe li prumire ataca-st a shofler di tote si foice, mins å pus k' ele sofléve, å pus ki l' voyaedjeu si ratoirtyive dins s' mantea ; eyet al fén, ele rinonça-st a l' disvôtyî. Adon c' est l' solo k' ataca-st a loumer, ey åd bout d' on moumint, li voyaedjeu, bén eschandi, oista s' mantea. Ça fwait ki l' bijhe diva bén ricnoxhe ki c' esteut l' solea li pus foirt di zels deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/liege.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe et l' solê estint e margaye. Et tchik et tchak, et tchik et tchak : one paskeye ki n' éve pont d' fin, dji vos di. Paski chakin d' zels dijéve k' il astéve li pus foirt. Mins on bê djoû, vola k' i rescontrèt on voyadjeû k' arivéve pal trin d' Mårloye et k' a dischindu al gåre di Jmèle. Mondiu todi ! Kén apoticaire ! K' il éve må l' air avou ses tchvès emacralès ! Il estéve droldimint agayolè, tot ravôtiè dins on vî paltot trawè pa les mites. On vrai bribeû ! Mètans nos d' acoird ! dit-st i onk des deus lascår. Li prèmî k' arive a foirci l' ome si Mårloye a rsatchè s' paltot, serè rwaitè come on vrai castår. Adon, a ! kéne arèdje, mes djins ! Li bijhe a cmincè a chuflè di totes ses foices, et a stårè tos les cantias d' Celina. Maria deyî ! Kéne atelêye vaici ! Mins dipus k' ele chufléve, dipus ki nost ome tronnéve et s' racrapotéve dins s' cazake. Al fin des fins, li bijhe, tote disbåtchée, a leyî tumer les brès. A ! c' est pol diåle, dit-st ele. Dji n' ai nin plu lî fé rôster s' paltot, nodidjo! Nosse solê ni fiéve pont d' bin. Mins i n' estéve nin biesse, et il end a profitè po cmincè a lûre, a lûre télmint k' il a mètu ses nwârès berikes po n' nin yesse li-minme asbleuwi. One miète pus tård, l' ome di Mårloye k' éve ossi tchôd k' dins on for, a fini pa s' dismoussè. Ça fait insi ki l' bijhe, elle a stî foircée d' ricnoxhe, ki l' solê éve gangnè, et ki ç' grandiveûs la estéve vraimint l' pus foirt.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/marche_en_famenne.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe eyet l' solea s' bretine, tchaconk acertinant k' il esteuve li pus foirt, cwand il ont voeyu l' voyaedjeu avancî, ewalpé dins s' paltot. I s' ont metou d' acoird kel cénk k' arivreuve li prumî a fé oister s' paltot å voyaedjeu sreut rmetou come li pus foirt. Adon l' bijhe s' a metou a shofler di tote foice mins dpus k' ele shofléve, dipus kel voyaedjeu seréve si paltot åtoû d' lu et, al difén, l' bijhe a rnoncî a l' lyi fé oister. Ça fwait kel solea a-st ataké a rlure et après on moumint, l' voyaedjeu, restchåfé, a oisté s' paltot. C' est insi kel bijhe a dvou rconexhe kel solea esteuve li pus foirt di zels deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/mont_saint_guibert.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li vint d' bijhe et l' solea si margayént, si tchaeke di zels dire ki c' est li l' pus foirt. So ç' trevén la, il ont veyou on voyaedjeu k' esteuve rotant, ewalpé dins s' cote. Il ont bouxhî djus : li prumî d' zels ambedeus ki stitchrè dins l' tiesse do voyaedjeu d' bouter s' cote djus, et k' l' ome si laireuve adire, c' est çti-la ki serè l' pus foirt. Aprume, li vint d' bijhe s' a metou a shofler di totes ses foices, mins todi pus k' i shofléve todi pus l' ome seréve si mantea åtoû d' lu, et po fini, li vint a leyî ouve. Adon, li solea a ataké a rglati et, pitchote a midjote, l' ome s' a tchåfé, et rtchåfé, et ristchåfe tu, disca tant k' il a tapé s' camizole djus. Sifwaitmint, li vint d' bijhe a rconoxhou ki l' solea esteuve li pus foirt did zels ambedeus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/namur.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("Li bijhe èt l' solê si cpetrognint, tchake acertinant k' il esteut li pus foirt. Cand ils ont vèyu on vwayadjeu k' ariveut, ravôtyi dins on paltot, i s' ont mètu dacoird po dire ki li ci ki parvinreut a lyi fé rsatchi si mousmint, sèreut vèyu come li pus foirt. Li bijhe s' a mètu a soflè a rlagne, di totes ses foices. Èt rin a fè! Dipus çk' èle soflèt, dipus çki l' ome rissèrèt s' paltot åtoû d' li. Èt po fini, li bijhe a fait hôw èt abandnè. Adon, li solê a cminci a rglati, èt après one hapée, l' ome, richandi, a rtirè si paltot. Do côp l' bijhe a bin du rconuche ki l' solê esteut li pus foirt di zels deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/rochefort.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)

stn_tst = get_text("C' esteut l' bijhe et l' solea k' estént higne et hagne. Paski pocwè? Paski tchaeke di zels forbateut k' il esteut l' pus foirt. A on metou moumint, la k' il avizèt on vî pôve ki s' apoentéve tot doûçmint. L' ome s' aveut racamizolé dins on grand lådje caban d' cur. Adon, i tuzèt a sayî, tchaeke a s' toû, di lyi fé saetchî s' paltot; et bouxhî l' martchî djus ki l' ci u l' cene k' adierceyreut s' côp sereut relî come li pus foirt. C' est l' bijhe ki cmince, ca on lait todi les cmeres fé les prumires. Et vo l' la ki s' mete a shofler et rashofler, et shofele mu co. Mins brosse di gade ! Pus çk' ele shofléve, pus ki l' vî rôlî rihaetchive si capote après lu, tantea ki: nosse damabonde a dvou leyî ouve. C' esteut l' toû do solea. Il atake a rlure, et rglati come cwand on creve di tchôd å cwénze d' awousse. Ça fwait k' nosse pormoenneu a stî reschandi; si a-t i hiné s' paltot djus. Ebén dabôrd, aiwdirotchmint, c' est l' solea k' a stî ricnoxhou li pus foirt di zels deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/transinne2.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)



