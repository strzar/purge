#!/bin/bash

#SBATCH --partition=h200_only
#SBATCH --nodelist=hctlrds2

#SBATCH --job-name=purge
#SBATCH --output=../logs/%x-%j.log
#SBATCH --error=../logs/%x-%j.log

#SBATCH --tasks=1           
#SBATCH --time=48:00:00     ## Maximum Runtime (HH:MM:SS, default 48:00:00)
#SBATCH --gres=gpu:h200:1   ## GPUs (amount, default 0)
#SBATCH --mem=100G          ## RAM (default 8GB)
#SBATCH --cpus-per-task=4   ## Cores (default 2)

##-------------------------------------------------------------------------------
names=('1_Stephen_King' '2_Confucius' '3_Bruce_Lee' '4_Warren_Buffett' '5_Christina_Aguilera')
# '1_Stephen_King' '2_Confucius' '3_Bruce_Lee' '4_Warren_Buffett' '5_Christina_Aguilera'
# '6_Cindy_Crawford' '7_Marie_Osmond' '8_Paris_Hilton' '9_Justin_Bieber' '10_Prince_Harry,_Duke_of_Sussex'
# '11_Miley_Cyrus' '12_Genghis_Khan' '13_Liza_Minnelli' '14_Taylor_Swift' '15_Mark_Cuban'
# '16_Rhea_Perlman' '17_Mark_Hamill' '18_John_D._Rockefeller' '19_Alanis_Morissette' '20_Marlon_Brando'
# '21_50_Cent' '22_Jim_Morrison' '23_Evel_Knievel' '24_Beyoncé' '25_Reba_McEntire'
# '26_Justin_Timberlake' '27_Vanna_White' '28_Lil_Wayne' '29_Anna_Nicole_Smith' '30_Henry_Winkler'
# '31_Leonardo_da_Vinci' '32_Kanye_West' '33_Paul_Walker' '34_Daniel_Day-Lewis' '35_Jim_Parsons'
# '36_Henry_Kissinger' '37_Chuck_Norris' '38_Steven_Seagal' '39_Linda_Hamilton' '40_Danny_Trejo'
# '41_Sam_Elliott' '42_Michael_Strahan' '43_Paul_Simon' '44_Meghan,_Duchess_of_Sussex' '45_Bruce_Springsteen'
# '46_Raquel_Welch' '47_Lenny_Kravitz' '48_Bob_Saget' '49_Jon_Voight' '50_Ryan_Seacrest'
# '51_Betty_White' '52_Chris_Brown' '53_Travis_Kelce' '54_Jay-Z' '55_Jackie_Chan'
# '56_Mark_Harmon' '57_Whitney_Houston' '58_Rihanna' '59_Anderson_Cooper' '60_Brendan_Fraser'
# '61_Tim_Burton' '62_Serena_Williams' '63_Dionne_Warwick' '64_Michelle_Pfeiffer' '65_Selena_Gomez'
# '66_Kris_Jenner' '67_Hugh_Laurie' '68_Tom_Clancy' '69_John_Candy' '70_Vin_Diesel'
# '71_Dakota_Fanning' '72_R._Kelly' '73_Emilio_Estevez' '74_Socrates' '75_Brooke_Shields'
# '76_Bob_Barker' '77_Val_Kilmer' '78_Jennifer_Lopez' '79_Pamela_Anderson' '80_Tony_Blair'
# '81_Vincent_van_Gogh' '82_Lindsay_Lohan' '83_Rebel_Wilson' '84_Nicolas_Cage' '85_Ted_Danson'
# '86_John_Travolta' '87_Robert_Downey_Jr.' '88_Jason_Bateman' '89_Samuel_L._Jackson' '90_Karl_Marx'
# '91_Halle_Berry' '92_Larry_Bird' '93_Johnny_Cash' '94_Chevy_Chase' '95_Bill_Paxton'
# '96_Ice_Cube' '97_Don_Johnson' '98_Dwayne_Johnson' '99_RuPaul' '100_Matthew_Perry'

for target in "${names[@]}"; do
    echo "============================================================"
    echo "Starting unlearning for: $target"

    python ../src/purge.py entity=$target

    echo "Finished unlearning for: $target"
    echo "============================================================"
done