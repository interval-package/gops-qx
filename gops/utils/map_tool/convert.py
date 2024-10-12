from lib.map2 import Map2

# MAP_NAME = "apollo_zxz"
# MAP_NAME = "ndgx_jsdd"
# MAP_NAME = "JinFengdadao"
MAP_NAME = "apollo_zxz_include_stoplines"
# MAP_NAME = "金山大道_时区49_修改偏置"


def main():
    map = Map2(MAP_NAME)
    map.load("./map/{}.json".format(MAP_NAME))
    map.run()
    map.match_stop_line()
    map.save("./map/{}_attach.json".format(MAP_NAME))


if __name__ == "__main__":
    main()
