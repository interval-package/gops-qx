from lib.map import Map

SHOW_LINK_BOUNDARY = False
SHOW_ID = True
SHOW_STOPLINE = True

# MAP_NAME = "JinFengdadao"
MAP_NAME = "crossroads_map"


def draw():
    map = Map()

    # Load new maps
    map.load_new("/root/qianxing/gops-grpc/gops/utils/map_tool/lib/map/crossroads_map_refined.json")

    # Load old maps
    map.load("/root/qianxing/gops-grpc/gops/utils/map_tool/lib/map/crossroads_map_refined.json")
    # map.load("./map/{}.json".format(MAP_NAME))

    map.draw(show_id=SHOW_ID, show_link_boundary=SHOW_LINK_BOUNDARY, map_name=MAP_NAME)


if __name__ == "__main__":
    draw()
