# Geographical data aggregation
## Goal
> Some observational data is represented as a set of shapes with a value assigned to each shape. An example are clouds of smoke caused by wildfires. These clouds are registered by satellites and each cloud is assigned a value for density of the smoke. Additionally, other data can be included, such as the time period for the observation, i.e. time period when the cloud was visible. We will call this set of shapes observational set, or observational shapefile.  
> For data processing pipelines it is often required to aggregate this data by geographic regions, that are also represented as a set of shapes. Geographic region shapefile can be contain counties, postal codes, countries, states, etc. We will call this set geographic set of geographic shapefile.  
> The idea is to look what observational data from the observational shapefile exists for each given region for a given period of time and calculate aggregate value for each region and time period. For example calculate aggregated value of smoke density for each postal code area.  
> <cite>[Michael Bouzinier]</cite>

## Types of aggregation
Two possible types of aggregation are implemented which are called "MAX" and "AVG_AREA_WEIGHTED". 
Also, there is an additional option of aggregation over time (by year, by month or by day).  
Here is an explanation how the aggregation works with time option.
### MAX
If no time aggregation type is specified maximum value of obs-id column from obs-shp is returned for every area in geo-shp file.  
If time aggregation type is specified then every observation in obs-shp file is preliminarily annotated with period of data 
during which this event occurred. And then for every area in geo-shp file and every period of time observed maximum value 
of obs-id column from obs-shp is returned. 
### AVG_AREA_WEIGHTED
If no time aggregation type is specified for every area in geo-shp file value of obs-id column from obs-shp file 
averaged by area that event occupied is returned (without weighting by fraction of time period occupation).  
If time aggregation type is specified then every observation in obs-shp file is preliminarily annotated with period of data 
during which this event occurred, and resulting value for every area in geo-shp is additionally weighted by 
fraction of time period during which the event occurred. 

## Script usage
```
usage: aggregator.py [-h] [--geo-shp GEO_SHP] [--geo-id GEO_ID]
                     [--geo-add GEO_ADD [GEO_ADD ...]] [--obs-shp OBS_SHP]
                     [--obs-id OBS_ID [OBS_ID ...]] [--agg-type AGG_TYPE]
                     [--epsg EPSG] [--time-agg TIME_AGG]
                     [--time-start-id TIME_START_ID]
                     [--time-end-id TIME_END_ID]
                     [--time-id-format TIME_ID_FORMAT] [--out-csv OUT_CSV]

Spatial data aggregation. Attention! In this code some technical columns are
added to datasets with names starting from "aggregator_". Do not use such
column names in input files to avoid interference!

options:
  -h, --help            show this help message and exit

Geographical shape file:
  --geo-shp GEO_SHP     Path to geographical .shp file
  --geo-id GEO_ID       The name of the identifier column of the geographic
                        region
  --geo-add GEO_ADD [GEO_ADD ...]
                        A set of additional columns (fields, annotations) from
                        geographic shapes to include in the result

Observational shape file:
  --obs-shp OBS_SHP     Path to observational .shp file
  --obs-id OBS_ID [OBS_ID ...]
                        The name(s) of the value column for observational
                        shapes

Aggregation parameters:
  --agg-type AGG_TYPE [Default: 5070]   Aggregation type: MAX|AVG_AREA_WEIGHTED
  --epsg EPSG           Which EPSG to use when counting area

Time aggregation parameters:
  --time-agg TIME_AGG   Time aggregation type if required:
                        DAILY|MONTHLY|ANNUAL
  --time-start-id TIME_START_ID [Default: Start]
                        The name of the datetime column of observation start
                        (for time aggregation)
  --time-end-id TIME_END_ID [Default: End]
                        The name of the datetime column of observation end
                        (for time aggregation)
  --time-id-format TIME_ID_FORMAT [Default: %Y%j %H%M]
                        The datetime format string of columns 'time-start-id'
                        and 'time-end-id' (to parse with datetime.strptime)

Output options:
  --out-csv OUT_CSV     Path to output aggregated data .csv
```
## Script launching example
```
python aggregator.py --obs-shp data_examples/observation_shape_sample/shapefile_21_22.sub.shp \
    --geo-shp data_examples/geography_shape/cb_2018_us_county_500k/cb_2018_us_county_500k.shp \
    --geo-id COUNTYNS \
    --geo-add NAME \
    --agg-type AVG_AREA_WEIGHTED \
    --time-agg ANNUAL \
    --obs-id Density \
    --out-csv hms_smoke_21_12_annual_by_cb_2018_us_county_500k.csv
```