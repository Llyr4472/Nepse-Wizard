import json
import requests
from datetime import datetime

base_url = "https://the-value-crew.github.io/nepse-api"
COMPANIES = ['HRL', 'SPDL', 'GIBF1', 'NGPL', 'SRLI', 'HLBSL', 'FMDBL', 'ICFC', 'GUFL', 'SBL', 'HDL', 'RADHI', 'TPC', 'SMJC', 'RBCLPO', 'RBBD83', 'BPCL', 'KBL', 'NESDO', 'SIKLES', 'MLBL', 'SEF', 'NBL', 'NRN', 'HBL', 'KSBBL', 'SAPDBL', 'CIT', 'SLBBL', 'RAWA', 'CMF2', 'KDBY', 'BNHC', 'SPIL', 'BNT', 'MERO', 'NYADI', 'SAHAS', 'MHCL', 'RURU', 'GBIME', 'HIDCL', 'NLICL', 'LEC', 'EDBL', 'BEDC', 'GLH', 'SAGF', 'SBI', 'TSHL', 'CBBL', 'SAMAJ', 'JSLBB', 'BFC', 'ALICL', 'SMB', 'CCBD88', 'UPCL', 'BARUN', 'MHL', 'MPFL', 'TAMOR', 'UMHL', 'GBLBS', 'SWBBL', 'NBLD87', 'NIFRA', 'HLI', 'PRSF', 'NFS', 'PBLD87', 'SMH', 'TRH', 'SHEL', 'SRBLD83', 'SNLI', 'MSLB', 'KSBBLD87', 'MBL', 'SLCF', 'SCB', 'SHIVM', 'SANIMA', 'SADBL', 'GRDBL', 'PROFL', 'SJCL', 'NMFBS', 'SHL', 'NABIL', 'RFPL', 'PMHPL', 'SBCF', 'SFEF', 'SFCL', 'SINDU', 'CKHL', 'LBBLD89', 'CBLD88', 'KKHC', 'HHL', 'MLBBL', 'OHL', 'HEI', 'MKHC', 'NIBLGF', 'PCBL', 'KDL', 'NADEP', 'FOWAD', 'MEN', 'SAEF', 'KPCL', 'SONA', 'CHL', 'PCBLP', 'NICD83/84', 'NIL', 'KMCDB', 'CHCL', 'LUK', 'MLBS', 'RNLI', 'CGH', 'GBILD86/87', 'MMKJL', 'LLBS', 'SALICO', 'CMF1', 'IHL', 'CZBIL', 'GMFBS', 'AVYAN', 'EBL', 'UNHPL', 'MAKAR', 'UPPER', 'GBBL', 'RLFL', 'ACLBSL', 'LBBL', 'RBCL', 'CHDC', 'HDHPC', 'MANDU', 'HATHY', 'NIMBPO', 'UNLB', 'NIMB', 'SPL', 'STC', 'RSDC', 'NICSF', 'HPPL', 'PSF', 'RIDI', 'NHPC', 'WNLB', 'NRIC', 'DHPL', 'RHGCL', 'NBF3', 'DDBL', 'RMF2', 'CYCL', 'AKJCL', 'GVL', 'NIBSF2', 'SBLD84', 'MSHL', 'SHPC', 'AKPL', 'ILBS', 'PHCL', 'JALPA', 'CIZBD90', 'AHPC', 'USLB', 'NMBMF', 'PRIN', 'MHNL', 'PPL', 'DLBS', 'BNL', 'NSIF2', 'GCIL', 'EHPL', 'SMATA', 'RHPL', 'SABSL', 'MLBSL', 'SIGS3', 'MKLB', 'MFIL', 'MCHL', 'MKHL', 'NICAD85/86', 'BHDC', 'NICGF2', 'NWCL', 'GMFIL', 'NHDL', 'NMB', 'NICGF', 'BGWT', 'SMFBS', 'NICBF', 'KSY', 'SSHL', 'CFCL', 'SPC', 'DOLTI', 'HURJA', 'PRVU', 'LICN', 'KBSH', 'CORBL', 'PMLI', 'GLBSL', 'KEF', 'NRM', 'SWMF', 'BHL', 'MKJC', 'NTC', 'BOKD86', 'MFLD85', 'SJLIC', 'NICA', 'NUBL', 'NBF2', 'NBLD82', 'C30MF', 'SARBTM', 'ADBL', 'ALBSL', 'ULHC', 'USHL', 'JFL', 'SGHC', 'VLBS', 'KRBL', 'KBLD89', 'PFL', 'API', 'PPCL', 'SGIC', 'UHEWA', 'SFMF', 'TVCL', 'BBC', 'NICLBSL', 'GILB', 'SIGS2', 'KLBSL', 'SDLBSL', 'CITY', 'LVF2', 'NLIC', 'NBLD85', 'SKBBL', 'NICFC', 'PBLD86', 'HIDCLP', 'UAIL', 'ULBSL', 'VLUCL', 'AHL', 'GHL', 'RMF1', 'JBBL', 'USHEC', 'DORDI', 'SMHL', 'NMB50', 'JOSHI', 'SHINE', 'MNBBL', 'MMF1', 'H8020', 'BHPL', 'GFCL', 'ILI', 'IGI', 'ENL', 'MEL', 'JBLB', 'LSL', 'NLG', 'SPHL', 'SHLB', 'MEHL', 'MKCL', 'ANLB', 'EBLD85', 'ADBLD83', 'HEIP', 'UMRH', 'JBBD87', 'NICL', 'MDB', 'MBJC', 'SICL', 'SIFC', 'NABBC']
with open("data/stocks.json",'r') as f:
    info = json.load(f)


class Stock:
    """
    Represents a stock with its symbol, ID, name, and trade information.

    The `Stock` class provides methods to retrieve historical data for the stock within a specified date range.

    Args:
        symbol (str): The stock symbol.

    Attributes:
        symbol (str): The stock symbol in uppercase.
        id (str): The unique identifier for the stock.
        name (str): The name of the stock.
        trade (dict): The trade information for the stock.
        file (str): The file path for the stock's historical data.

    Methods:
        __call__(): Returns the stock symbol.
        __eq__(other): Compares two `Stock` objects for equality based on their ID.
        get_data(start_date=None, end_date=None): Retrieves the historical data for the stock within the specified date range.
    """

    def __init__(self,symbol):
        self.symbol = symbol.upper()
        self.id = info[self.symbol]['id']
        self.name = info[self.symbol]['name']
        self.trade = info[self.symbol]['trade']
        self.file = f"data/stocks/{self.symbol.replace('/','∕')}.csv"
    
    def __call__(self):
        return self.symbol

    def __eq__(self, other):
        if isinstance(other, Stock):
            return self.id == other.id
        return False
    
    def get_data(self,start_date=None,end_date=None):
        endpoint = f"/data/company/{self.symbol.replace('/','∕')}.json"
        data = request_json(endpoint)
        
        if not start_date and not end_date:
            return data
        
        if not end_date:
            if not isinstance(start_date, datetime):start_date = datetime.strptime(start_date, '%Y-%m-%d')
            return {date: values for date, values in data.items() if start_date <= datetime.strptime(date, '%Y-%m-%d')}

        if not start_date:
            if not isinstance(end_date, datetime):end_date = datetime.strptime(end_date, '%Y-%m-%d')
            return {date: values for date, values in data.items() if datetime.strptime(date, '%Y-%m-%d') <= end_date}

        if not isinstance(start_date, datetime):start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if not isinstance(end_date, datetime):end_date = datetime.strptime(end_date, '%Y-%m-%d')
        return {date: values for date, values in data.items() if start_date <= datetime.strptime(date, '%Y-%m-%d') <= end_date}

def request_json(endpoint):
    url = base_url + endpoint
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    print("Error", response.status_code, f" in {url}")
    return {}

def dump(data):
    with open("dump.json",'w') as f:
        json.dump(data,f)

if __name__ == '__main__':
    endpoint = '/data/companies.json'
    data = request_json(endpoint)
    print(data)
