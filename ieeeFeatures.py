'''
Feature Engineering for Kaggle Vesta Fraud detection

'''

## Combines Device information for devices
def ft_deviceInfo(data):
    # DeviceInfo: Try to group devices by brand
    DevInfo = data['DeviceInfo']

    samsung_devs = DevInfo[DevInfo.str.startswith(('SM','SAMSUNG')).fillna(False)].unique()
    huawei_devs = DevInfo[DevInfo.str.contains('Huawei|hi6210sft', case=False).fillna(False)].unique()
    motorola_devs = DevInfo[DevInfo.str.contains('Moto', case=False).fillna(False)].unique()
    windows_devs = DevInfo[DevInfo.str.startswith(('rv','Trident')).fillna(False)].unique()
    lg_devs = DevInfo[DevInfo.str.startswith('LG').fillna(False)].unique()
    apple_devs = DevInfo[DevInfo.str.startswith(('iOS','MacOS')).fillna(False)].unique()
    linux_devs = DevInfo[DevInfo.str.startswith('Linux').fillna(False)].unique()

    DevInfo = DevInfo.replace(samsung_devs,'SAMSUNG')
    DevInfo = DevInfo.replace(huawei_devs,'HUAWEI')
    DevInfo = DevInfo.replace(motorola_devs,'MOTOROLA')
    DevInfo = DevInfo.replace(windows_devs,'Windows')
    DevInfo = DevInfo.replace(lg_devs,'LG')
    DevInfo = DevInfo.replace(apple_devs,'APPLE')
    DevInfo = DevInfo.replace(linux_devs,'LINUX')

    top_devs = ['Windows','APPLE','SAMSUNG','MOTOROLA','HUAWEI','LG','LINUX']
    # For now, just label the remaining lower frequencies as 'other'
    DevInfo = DevInfo.replace(DevInfo.loc[~DevInfo.isin(top_devs) & \
     ~DevInfo.isnull()].unique(),'OTHER')

    # Return DeviceInfo Data
    return(DevInfo)


## Groups OS systems

def ft_os(data):
    osInfo = data['id_30']
    windows_os = osInfo[osInfo.str.startswith('Windows').fillna(False)].unique()
    ios_os = osInfo[osInfo.str.startswith('iOS').fillna(False)].unique()
    mac_os = osInfo[osInfo.str.startswith('Mac').fillna(False)].unique()
    android_os = osInfo[osInfo.str.startswith('Android').fillna(False)].unique()

    osInfo = osInfo.replace(windows_os,'Windows')
    osInfo = osInfo.replace(ios_os,'iOS')
    osInfo = osInfo.replace(mac_os,'Mac')
    osInfo = osInfo.replace(android_os,'Android')

    return(osInfo)


## Groups Browsers
def ft_browser(data):
    browserInfo = data['id_31']

    safari_browser = browserInfo[browserInfo.str.contains('safari', case=False).fillna(False)].unique()
    chrome_browser = browserInfo[browserInfo.str.contains('chrome', case=False).fillna(False)].unique()
    firefox_browser = browserInfo[browserInfo.str.contains('firefox', case=False).fillna(False)].unique()
    ie_browser = browserInfo[browserInfo.str.contains('ie', case=False).fillna(False)].unique()
    edge_browser = browserInfo[browserInfo.str.contains('edge', case=False).fillna(False)].unique()
    samsung_browser = browserInfo[browserInfo.str.contains('samsung', case=False).fillna(False)].unique()


    browserInfo = browserInfo.replace(safari_browser,'safari')
    browserInfo = browserInfo.replace(chrome_browser,'chrome')
    browserInfo = browserInfo.replace(firefox_browser,'firefox')
    browserInfo = browserInfo.replace(ie_browser,'ie')
    browserInfo = browserInfo.replace(edge_browser,'edge')
    browserInfo = browserInfo.replace(samsung_browser,'samsung')

    return(browserInfo)

## Creates a date based off of the Dec 1, 2017 start date
def ft_createDate(data):
    import datetime
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    date = data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
    return(date)
