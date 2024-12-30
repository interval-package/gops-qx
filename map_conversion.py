def covert_map(self, traffic_map):
        link_index = 0  
        lane_index = 0

        for index, segment in enumerate(traffic_map.data.segments):
            self.segment2idx[segment.id] = index
            self.segment_id_list.append(segment.segment_id)

            for link in segment.ordered_links:
                self.link2idx[link.id] = link_index
                link_index += 1
                self.link_id_list.append(link.id)
                
                for lane in link.ordered_lanes:
                    self.lane2idx[lane.id] = lane_index
                    lane_index += 1
                    self.lane_id_list.append(lane.id)
        
        for index,junction in enumerate(traffic_map.data.junctions):
            self.junction2idx[junction.id] = index
            self.junction_id_list.append(junction.id)