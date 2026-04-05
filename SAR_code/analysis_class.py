import asf_search as asf
import pandas as pd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
from dateutil.parser import parse
from datetime import timedelta
from typing import Optional,Union,List,Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
import os
import ast


class Analyzer:
    def __init__(
        self,
        data_frame:Union[pd.DataFrame,str,None],
        product_column_name:Union[str,None]=None,
        product_column_index:int = False,
        Esa_product_name:Union[str,None] = None
    ):
        self.data_frame = data_frame if isinstance(data_frame,pd.DataFrame) else pd.read_csv(str(data_frame))
        self.Esa_product_name = Esa_product_name
        self.Esa_products = list(self.data_frame[product_column_name] if isinstance(product_column_name,str) else self.data_frame[list(self.data_frame)[product_column_index]])
        # print(self.Esa_products)
        self.single_product_info = self.Search_by_name(self.Esa_product_name)[0] if isinstance(self.Esa_product_name,str) else None
        self.list_product_info = self.Search_by_name(self.Esa_products) 

        self.scene_info_dictionary = self.build_Scene_index()
        self.FramePath_info_dictionary = self.build_FramePath_index()

    #  Visulazing the overlaps and patches
    def Overlap(self,df:pd.DataFrame):
        
        return None
    
    def small_check(self):
        SceneName_list = []
        for result in self.list_product_info:
            if result is not None:
                scene_name = result.properties["sceneName"]
                if scene_name not in self.Esa_products:
                    print("Sorry this product was given")
                    break
                SceneName_list.append(scene_name)
        
        if len(SceneName_list) == len(self.Esa_products):
            print("Scene are same")
        else:
            print("Scenes are not same")
    
    # This function will give the list of info we wanted from the result
    def create_Listproperty(self,result,n_property):
        if result is not None:
            property_list = []
            for r in result:
                property_list.append(r.properties[n_property])
            return property_list
        else:
            raise ValueError("Some problem in Result")
        
    def create_Listgeometry(self,result):
        if result is not None:
            geometry_list = []
            for r in result:
                geometry_list.append(r.geometry)
            return geometry_list
        else:
            raise ValueError("Some problem in Result")

    # This function will group all the product that are provided to us using the same path and frame
    def Group_given_location_and_get_count(self,print_info:bool = False):
        print(len(self.list_product_info))
        Properties = {}
        Geometry = {}

        SceneName_list = []
        path_list = []
        frame_list = []
        direction_list = []
        Coordinate_list = []
        startTime_list = []
        stopTime_list = []
        for index,result in enumerate(self.list_product_info):
            if result is not None:
                Properties = result.properties
                Geometry   = result.geometry
                SceneName_list.append(Properties["sceneName"])
                path_list.append(Properties["pathNumber"])
                frame_list.append(Properties["frameNumber"])
                direction_list.append(Properties["flightDirection"])
                startTime_list.append(Properties["startTime"])
                stopTime_list.append(Properties["stopTime"])
                Coordinate_list.append(Geometry)
            else:
                print("All scenes result were not available few are missing")

            # print(f"indexv: {index}| Scene : {Properties['sceneName']}  |Coordinates: {Geometry}")
        analysis_df = self.create_dataframe(
            ['sceneName','pathNumber','frameNumber','Direction','startTime','stopTime','Coordinate&Type'],
            SceneName_list,path_list,frame_list,
            direction_list,startTime_list,stopTime_list,
            Coordinate_list
        )
        print(analysis_df.iloc[:50])
        print("-"*50)
        analysis_df_sort = analysis_df.sort_values(by=['pathNumber','frameNumber','Direction'])
        
        if print_info:
            stacks = analysis_df.groupby(['pathNumber','frameNumber','Direction'])
            for (path,frame,direction),stack_df in stacks:
                stack_df['startTime'] = pd.to_datetime(stack_df['startTime'])
                print(f"Stack:Path {path} | Frame:{frame} | Direction:{direction}")
                print(f"Count:{len(stack_df)} scenes")
                
                start = stack_df['startTime'].min()
                end = stack_df['startTime'].max()
                duration = (end - start).days
                print(f"Timeline:{start.date()} to {end.date()} ({duration} days)")
                print("-"*50)
        else:
            print(analysis_df_sort.iloc[:50])
    
    def Group_by_scene(self,target_sceneName):
        scene_index = {r.properties["sceneName"]:r for r in self.list_product_info}
        scene_result = scene_index.get(target_sceneName)
        if scene_result is not None:
            path = scene_result.properties["pathNumber"]
            frame = scene_result.properties["frameNumber"]
            product_list = self.Group_by_FramePath(path,frame)
            return product_list
        return None
    
    def build_Scene_index(self):
        Scene_dict = {}
        for r in self.list_product_info:
            props = r.properties
            key = props.get("sceneName")
            Scene_dict[key] = r
        return Scene_dict

    def build_FramePath_index(self):
        scene_index = defaultdict(list)
        for r in self.list_product_info:
            props = r.properties
            key = (props.get("pathNumber"),props.get("frameNumber"))
            scene_index[key].append(r)
        return scene_index
    
    def get_by_pathframe(self,target_pathNumber,target_frameNumber):
        Scene_Index = self.FramePath_info_dictionary
        return Scene_Index.get((target_pathNumber,target_frameNumber),[])

    def Group_by_FramePath(self,target_pathNumber,target_frameNumber):
        result_list = self.get_by_pathframe(target_pathNumber=target_pathNumber,target_frameNumber=target_frameNumber)
        return result_list
    
    # This function will collect all scenes and patches that overlap with given data
    def Overlap_patches(self,target_filename=None,target_search_start=None,target_search_end=None,target_sceneName=None,target_pathNumber=None,target_frameNumber=None):
        Start_time = f"{target_search_start}T00:00:00Z"
        End_time = f"{target_search_end}T23:59:59Z"
        Intrest_Location = None

        if target_sceneName:

            if self.scene_info_dictionary[target_sceneName] is not None:
                result = self.scene_info_dictionary[target_sceneName]
                Intrest_Location = result.geometry
            # We need to search the location
            else:
                Information = asf.granule_search([target_sceneName])
                if len(Information)>0:
                    result = Information[0]
                    Intrest_Location = result.geometry
                else:
                    raise ValueError("The given scene does not exist in the database")
        else:
            Info_result = self.get_by_pathframe(target_pathNumber=target_pathNumber,target_frameNumber=target_frameNumber)
            if len(Info_result)>0:
                Needed_result = Info_result[0]
                Intrest_Location = Needed_result.geometry
            else:
                Info_result = self.fetch_Scene_from_FramePath_Date(target_pathNumber=target_pathNumber,target_frameNumber=target_frameNumber,target_Starttime=target_search_start,target_EndTime=target_search_end)
                if len(Info_result)>0:
                    Needed_result = Info_result[0]
                    Intrest_Location = Needed_result.geometry
                else:
                    raise ValueError("No results were found either check the time frame or frame and path number")
        print("Locations found.Searching for ovelaps..")
        #  Now we will implement the search logic
        if isinstance(Intrest_Location,str):           
            polygon_dict = ast.literal_eval(Intrest_Location)
        else:
            polygon_dict = Intrest_Location

        aoi_wkt = shape(polygon_dict).wkt
        result_geometry = asf.search(
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel=asf.PRODUCT_TYPE.GRD_HD,
            intersectsWith=aoi_wkt,
            start = Start_time,
            end = End_time
        )
        centerLat_list = self.create_Listproperty(result_geometry,"centerLat")
        centerLon_list = self.create_Listproperty(result_geometry,"centerLon")
        stopTime_list = self.create_Listproperty(result_geometry,"stopTime")
        startTime_list = self.create_Listproperty(result_geometry,"startTime")
        frameNumber_list = self.create_Listproperty(result_geometry,"frameNumber")
        pathNumber_list = self.create_Listproperty(result_geometry,"pathNumber")
        Location_list = self.create_Listgeometry(result_geometry)
        sceneNamme_list = self.create_Listproperty(result_geometry,"sceneName")
        flightDirection_list = self.create_Listproperty(result_geometry,"flightDirection")
        
        percentage_overlap_list = [self.Calculate_OverlapPercent(polygon_dict,location) for location in Location_list]
        Overlap_df = self.create_dataframe(
            ["sceneName","frameNumber","pathNumber","startTime","stopTime","flightDirection","centerLat","centerLon","Location","Overlap_%"],
            sceneNamme_list,frameNumber_list,
            pathNumber_list,startTime_list,
            stopTime_list,flightDirection_list,
            centerLat_list,centerLon_list,Location_list,
            percentage_overlap_list
        )
        self.save_to_csv(data=Overlap_df,filename=target_filename,show_path=True)

    # This will calculate how much area overlap in percentage
    def Calculate_OverlapPercent(self,base_coordinate,aoi_coordinate):
        base_polygon_dict = None
        aoi_polygon_dict = None

        if isinstance(base_coordinate,str):
            base_polygon_dict = ast.literal_eval(base_coordinate)
        else:
            base_polygon_dict = base_coordinate
        if isinstance(aoi_coordinate,str):
            aoi_polygon_dict = ast.literal_eval(aoi_coordinate)
        else:
            aoi_polygon_dict = aoi_coordinate
        
        base_geom = shape(base_polygon_dict)
        aoi_geom = shape(aoi_polygon_dict)
        gdf_base = gpd.GeoSeries([base_geom],crs="EPSG:4326")
        gdf_aoi = gpd.GeoSeries([aoi_geom],crs="EPSG:4326")

        gdf_base_metric = gdf_base.to_crs(epsg=3857)
        gdf_aoi_metric = gdf_aoi.to_crs(epsg=3857)
        base_metric_geom:BaseGeometry = gdf_base_metric.iloc[0] #type:ignore
        aoi_metric_geom:BaseGeometry = gdf_aoi_metric.iloc[0] #type:ignore

        intersection = base_metric_geom.intersection(aoi_metric_geom)

        intersect_area = intersection.area
        aoi_area = aoi_metric_geom.area

        if aoi_area==0:
            return 0.0
        overlap_pct = (intersect_area/aoi_area)*100
        return overlap_pct


    def GroupSame_location(self,filename=None,search_start=None,search_end=None,from_data=True,sceneName=None,pathNumber=None,frameNumber=None):
        # If you want to lacate from the given dataset than it is recommended that dont give the search dates
        if from_data:
            print("Thinking")
            if sceneName:
                results = self.Group_by_scene(sceneName)
                centerLat_list = self.create_Listproperty(results,"centerLat")
                centerLon_list = self.create_Listproperty(results,"centerLon")
                stopTime_list = self.create_Listproperty(results,"stopTime")
                startTime_list = self.create_Listproperty(results,"startTime")
                frameNumber_list = self.create_Listproperty(results,"frameNumber")
                pathNumber_list = self.create_Listproperty(results,"pathNumber")
                Location_list = self.create_Listgeometry(results)
                sceneName_list = self.create_Listproperty(results,"sceneName")
                flightDirection_list = self.create_Listproperty(results,"flightDirection")
                info_df = self.create_dataframe(
                    ["sceneName","frameNumber","pathNumber","startTime","stopTime","flightDirection","centerLat","centerLon","Location"],
                    sceneName_list,frameNumber_list,
                    pathNumber_list,startTime_list,
                    stopTime_list,flightDirection_list,
                    centerLat_list,centerLon_list,Location_list
                )
                self.save_to_csv(data = info_df,filename=filename,show_path=True)
            else:
                results_pf = self.Group_by_FramePath(target_frameNumber=frameNumber,target_pathNumber=pathNumber)
                centerLat_list = self.create_Listproperty(results_pf,"centerLat")
                centerLon_list = self.create_Listproperty(results_pf,"centerLon")
                stopTime_list = self.create_Listproperty(results_pf,"stopTime")
                startTime_list = self.create_Listproperty(results_pf,"startTime")
                frameNumber_list = self.create_Listproperty(results_pf,"frameNumber")
                pathNumber_list = self.create_Listproperty(results_pf,"pathNumber")
                Location_list = self.create_Listgeometry(results_pf)
                sceneName_list = self.create_Listproperty(results_pf,"sceneName")
                flightDirection_list = self.create_Listproperty(results_pf,"flightDirection")
                info_df_pf = self.create_dataframe(
                    ["sceneName","frameNumber","pathNumber","startTime","stopTime","flightDirection","centerLat","centerLon","Location"],
                    sceneName_list,frameNumber_list,
                    pathNumber_list,startTime_list,
                    stopTime_list,flightDirection_list,
                    centerLat_list,centerLon_list,Location_list
                )
                self.save_to_csv(data=info_df_pf,filename=filename,show_path=True)
        else:
            s_pathNumber = 0
            s_frameNumber = 0
            if sceneName:
                result = self.Search_by_name(sceneName)[0]
                if result is not None:
                    s_pathNumber = result.properties["pathNumber"]
                    s_frameNumber = result.properties["frameNumber"]
            else:
                s_pathNumber = pathNumber
                s_frameNumber = frameNumber

            Scene_found = self.fetch_Scene_from_FramePath_Date(
                target_pathNumber=s_pathNumber,
                target_frameNumber=s_frameNumber,
                target_Starttime=search_start,
                target_EndTime=search_end
                )
            centerLat_list = self.create_Listproperty(Scene_found,"centerLat")
            centerLon_list = self.create_Listproperty(Scene_found,"centerLon")
            stopTime_list = self.create_Listproperty(Scene_found,"stopTime")
            startTime_list = self.create_Listproperty(Scene_found,"startTime")
            frameNumber_list = self.create_Listproperty(Scene_found,"frameNumber")
            pathNumber_list = self.create_Listproperty(Scene_found,"pathNumber")
            Location_list = self.create_Listgeometry(Scene_found)
            sceneName_list = self.create_Listproperty(Scene_found,"sceneName")
            flightDirection_list = self.create_Listproperty(Scene_found,"flightDirection")
            info_df = self.create_dataframe(
                ["sceneName","frameNumber","pathNumber","startTime","stopTime","flightDirection","centerLat","centerLon","Location"],
                sceneName_list,frameNumber_list,
                pathNumber_list,startTime_list,
                stopTime_list,flightDirection_list,
                centerLat_list,centerLon_list,Location_list
            )
            self.save_to_csv(data=info_df,filename=filename,show_path=True)

            print("Ok")

    def fetch_Scene_from_FramePath_Date(self,target_pathNumber,target_frameNumber,target_Starttime,target_EndTime):
        start_time = f"{target_Starttime}T00:00:00Z"
        end_time = f"{target_EndTime}T23:59:59Z"
        path = target_pathNumber
        frame = target_frameNumber

        inventry_results = asf.search(
            platform=asf.PLATFORM.SENTINEL1,
            relativeOrbit=path,
            frame=frame,
            processingLevel=asf.PRODUCT_TYPE.GRD_HD,
            beamMode=asf.BEAMMODE.IW,
            start=start_time,
            end=end_time
        )
        if len(inventry_results) == 0:
            raise ValueError("Patch does not exist for the given configuration and time")
        return inventry_results
    
    def create_dataframe(self,columns,*lists):
        if len(columns) != len(lists):
            raise ValueError("Number columns mismatch")
        data = {col:lst for col,lst in zip(columns,lists)}
        return pd.DataFrame(data)
    
    def Create_dict(self,product_name)->Dict:
        field_info = product_name.split('_')
        information = {}
        information["platform"] = field_info[0]
        information["beam_mode"] = field_info[1]
        information["product_type"] = field_info[2]
        information["level"] = field_info[3][0]
        information["polarization"] = field_info[3][2:]
        information["Start_time"] = parse(field_info[4]) 
        information["End_time"] = parse(field_info[5])
        information["orbit"] = int(field_info[6])
        information["product_name"] = product_name
        return information


    def parse_sentinel1_product(self):
        product_info:List[Dict] = []
        if isinstance(self.Esa_product_name,str):
            print("Working on single")
            info_dict = self.Create_dict(self.Esa_product_name)
            product_info.append(info_dict)
        else:
            print("Working on List")
            for scene in self.Esa_products:
                info_dict = self.Create_dict(scene)
                product_info.append(info_dict)
        return product_info
    
    def Search_by_time(self,info_dictionary):
        dictionary = info_dictionary

        search_start = (dictionary["Start_time"] - timedelta(seconds=5)).isoformat()
        search_end = (dictionary["Start_time"] + timedelta(seconds=5)).isoformat()

        result = asf.search(
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel=asf.PRODUCT_TYPE.GRD_HD,
            start=search_start,
            end=search_end
        )

        if len(result) == 0:
            return None
        
        asf_item_found = result[0]
        return asf_item_found.properties,asf_item_found.geometry
    
    def Search_by_name(self, products):
        if isinstance(products, str):
            products = [products]

        products = list(set(products))

        results = asf.granule_search(products)
        seen = set()
        unique = []
        for r in results:
            scene = r.properties["sceneName"]
            if scene not in seen:
                seen.add(scene)
                unique.append(r)
        return unique
    
    def calculate_scene_area(self,geometry):
        if geometry is None:
            return None
        
        geom = shape(geometry)
        gdf = gpd.GeoDataFrame(
            {"geometry":[geom]},
            crs="EPSG:4326"
        )
        gdf_projected = gdf.to_crs(gdf.estimate_utm_crs())
        area_km2 = gdf_projected.area.iloc[0]/1e6
        return area_km2
    
    def update_csv_column(self,csv_file,mapping:dict,column_name:str):
        df = pd.read_csv(csv_file)
        if "sceneName" not in df.columns:
            raise ValueError("Csv must contain a 'sceneName' column")
        if column_name not in df.columns:
            df[column_name] = None
        
        df[column_name] = df["sceneName"].map(mapping).fillna(df[column_name])
        df.to_csv(csv_file,index=False)
        return df
    
    def save_to_csv(self,data,filename,show_path=False):
        output_dir = r"D:\SAR-Intelligence\data_engine\Analysis_output"
        os.makedirs(output_dir,exist_ok=True)

        file_path = os.path.join(output_dir,filename)
        if isinstance(data,pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
        
        df.to_csv(file_path,index=False)
        print(f"File Saved:{file_path}") if show_path == True else None


data_dir = r"D:\SAR-Intelligence\data_engine\SAR_Data_csv_info\ESA_xView3_sceneName_mapping.csv"
Exp_product_name = "S1B_IW_GRDH_1SDV_20200803T062012_20200803T062037_022755_02B2F6_ED2D"

test_scene = "S1A_IW_GRDH_1SDV_20201128T051123_20201128T051148_035444_042495_14D3"

# Analysis = Analyzer(data_dir,product_column_index=0,Esa_product_name=Exp_product_name)
# print(Analysis.single_product_info)

List_analysis = Analyzer(data_dir,product_column_index=0,Esa_product_name=None)

print(List_analysis.single_product_info)
print(len(List_analysis.list_product_info))

# print(List_analysis.scene_info_dictionary)

# List_analysis.small_check()

# List_analysis.Group_given_location()

# print(List_analysis.build_FramePath_index())

# print("-"*50)
# List_analysis.GroupSame_location(filename="Same_footprint_from_scene.csv",from_data=True,sceneName=test_scene)
# print("-"*50)
# List_analysis.GroupSame_location(filename="Same_footprint_from_pathframe.csv",from_data=True,frameNumber=446,pathNumber=22)
# print("-"*50)
# List_analysis.GroupSame_location(filename="Same_footprint_from_dataBase.csv",from_data=False,frameNumber=446,pathNumber=22,search_start='2020-01-01',search_end='2020-01-31')
# print("-"*50)
# List_analysis.Overlap_patches(target_filename="Same_footprint_Overlap_dataset_march_2020.csv",target_search_start='2020-03-01',target_search_end='2020-03-31',target_pathNumber=22,target_frameNumber=446)
# print("-"*50)

List_analysis.Group_given_location_and_get_count(print_info=True)

# list_res_info = List_analysis.parse_sentinel1_product()[0]
# result = List_analysis.Search_by_time(list_res_info)
# if result is not None:
#     pro,geo = result
#     print(list(pro.keys()))
#     print(list(geo.keys()))
#     print(f"Area of the scene:{List_analysis.calculate_scene_area(geo)}")