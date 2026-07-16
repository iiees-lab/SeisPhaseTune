from obspy.clients.filesystem.tsindex import Indexer
import os

root_path = r"D:/SeisComP_Archive"
database_path = r"D:/timeseries.sqlite"



# create a new Indexer instance
indexer = Indexer(
    root_path=root_path,
    database=r'D:/timeseries.sqlite',
    index_cmd=database_path,
    bulk_params=None,
    filename_pattern='*.D.*',
    parallel=5,
    leap_seconds_file=None,
    loglevel=None,
)

indexer.run()


##### Test
from obspy.clients.filesystem.tsindex import Client
from obspy import UTCDateTime
from obspy.clients.filesystem.tsindex import Indexer

root_path = r"D:/SeisComP_Archive"
database_path = r"D:/timeseries.sqlite"


# create a new Client instance
client = Client(
    database=database_path,
    datapath_replace=("/mnt/d/SeisComP_Archive", root_path)
)

# extents = client.get_availability_extent(network="AH", channel="HHZ")
# for extent in extents:
#     print("{0:<3} {1:<6} {2:<3} {3:<4} {4} {5}".format(*extent))

net = "AH"
sta = "AZGN"
loc = ""
cha = "HHZ"
stime = UTCDateTime("2012-08-14T10:49:00.000000Z")
etime = UTCDateTime("2012-08-15T09:50:00.990000Z")

# avail_percentage = client.get_availability_percentage(
#     net, sta, loc, cha,
#     stime,
#     etime
# )
# print(avail_percentage)

st = client.get_waveforms(net, sta, loc, cha, stime, etime)