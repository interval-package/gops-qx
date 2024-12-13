from lib.map3 import Map3

MAP_NAME = "crossroads_map"

def main():
    # 创建 Map3 实例
    map = Map3("./map/{}.json".format(MAP_NAME))
    
    # 加载地图数据
    map.load_map()
    
    # 填充缺失的链接ID
    map.fill_connection_ids()
    
    # 更新流向信息
    map.update_flow_directions()

    # match_stopline
    map.match_stopline()
    
    # 保存修改后的地图到新的JSON文件
    map.save_to_json("./map/{}_refined.json".format(MAP_NAME))

if __name__ == "__main__":
    main()
