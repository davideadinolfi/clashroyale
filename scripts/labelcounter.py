from pathlib import Path
from collections import Counter

def count_first_numbers(folder: str):
    counter = Counter()

    for txt_file in Path(folder).glob("*.txt"):
        with txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                first_token = line.split()[0]

                if first_token.isdigit():
                    counter[int(first_token)] += 1

    return counter


if __name__ == "__main__":
    folder = "../dataset/3/train/labels"  # <-- cambia con la cartella dei .txt
    counts = count_first_numbers(folder)
    labels=['EArcher', 'EArcherQueen', 'EArcherRascal', 'EBRam', 'EBabyDrag', 'EBalloon', 'EBandit', 'EBarb', 'EBat', 'EBeserker', 'EBigPekka', 'EBigRascal', 'EBomber', 'EBombertower', 'EBossBandit', 'EBowler', 'EBrawler', 'EBush', 'ECannonCart', 'ECannoneerTower', 'EDarkPrince', 'EDartGoblin', 'EDoctor', 'EEVArcher', 'EEVFirecrack', 'EEVFurnace', 'EEVLarry', 'EEVMegaKnight', 'EEVRoyalG', 'EEVTesla', 'EEVWitch', 'EEX', 'EElectroDrag', 'EElectroG', 'EEletrcoWizard', 'EEletroSpirit', 'EEliteBarb', 'EElixerGolem', 'EFireball', 'EFirecrack', 'EFirespirit', 'EFisherman', 'EFlyingMachine', 'EFurnace', 'EGiant', 'EGiantSkeleton', 'EGoblin', 'EGoblinBarrel', 'EGoblinCage', 'EGoblinGiant', 'EGoblinHut', 'EGoblindem', 'EGolem', 'EGolemMite', 'EGuard', 'EHealSpirit', 'EHog', 'EHogRider', 'EHunter', 'EIceGolem', 'EIceSpirit', 'EIceWizard', 'EInfernoDrag', 'EKingTower', 'EKnight', 'ELarry', 'ELavaHound', 'ELittlePrince', 'ELittlePrinceSide', 'ELumberJack', 'EMagicArcher', 'EMegaKnight', 'EMegaM', 'EMightyMiner', 'EMiner', 'EMiniPeka', 'EMonk', 'EMonster', 'EMorter', 'EMotherWitch', 'EMusk', 'EMuskNew', 'ENightWitch', 'EPhenix', 'EPrince', 'EPrincess', 'EPump', 'ERamRider', 'ERocket', 'ERoyalGhost', 'ERoyalHog', 'ERoyalKnight', 'ERoyalR', 'ERoyaleG', 'ESkelBarrel', 'ESnowBall', 'ESparky', 'ESpearGoblin', 'ESpiritEmp', 'ESpritEmpWalk', 'ETesla', 'ETomb', 'EValk', 'EWallBreak', 'EWitch', 'EWizard', 'EXbow', 'EZappy', 'Ecan', 'EnemyDagTower', 'EnemyPanTower', 'EnemyPrincessTower', 'EnemyRocket', 'OArcher', 'OBRam', 'OBabyDrag', 'OBalloon', 'OBandit', 'OBarb', 'OBat', 'OBeserker', 'OBigPekka', 'OBomber', 'OBombertower', 'OBowler', 'OBrawler', 'OCan', 'OCannoneerTower', 'ODarkPrince', 'ODartGoblin', 'ODoctor', 'OEVArcher', 'OEVElectroDrag', 'OEVFirecrack', 'OEVInfernoDrag', 'OEVRoyalG', 'OEVWizard', 'OElectroDrag', 'OElectroWizard', 'OElexirgolem', 'OEx', 'OFireSpirit', 'OFireWizard', 'OFirecrack', 'OFisherman', 'OFlyingMachine', 'OFurnace', 'OGiant', 'OGiantSkeleton', 'OGoblin', 'OGoblinBarrel', 'OGoblinCage', 'OGoblinDem', 'OGoblinDrill', 'OGoblinGiant', 'OGoblinHut', 'OGoblinMachine', 'OGolem', 'OGuard', 'OHealer', 'OHog', 'OHogRider', 'OHunter', 'OIceGolem', 'OIceSpirit', 'OIceWizard', 'OInfernoDrag', 'OInfernoTower', 'OKnight', 'OLarry', 'OLittlePrince', 'OLumberjack', 'OMagicArcher', 'OMegaKnight', 'OMegam', 'OMightyMiner', 'OMiniGolem', 'OMiniPekka', 'OMortar', 'OMotherWitch', 'OMusk', 'OMuskNew', 'OPrince', 'OPrincess', 'OPump', 'ORamrider', 'ORocket', 'ORoyalG', 'ORoyalGhost', 'ORoyalHog', 'ORoyalKnight', 'ORoyalR', 'OSkelDrag', 'OSparky', 'OSpearGoblin', 'OSpiritEmp', 'OTesla', 'OTomb', 'OValk', 'OWallBreak', 'OWitch', 'OWizard', 'OXbow', 'OZappy', 'OwnDagTower', 'OwnKingTower', 'OwnPanTower', 'OwnPrincessTower', 'TowerDown', 'eminion', 'ofireball', 'ominion']


    for key, value in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{labels[key]}: {value}")