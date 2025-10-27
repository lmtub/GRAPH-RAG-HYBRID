static uint16_t phys_map_node_alloc(void)

{

    unsigned i;

    uint16_t ret;



    ret = next_map.nodes_nb++;

    assert(ret != PHYS_MAP_NODE_NIL);

    assert(ret != next_map.nodes_nb_alloc);

    for (i = 0; i < L2_SIZE; ++i) {

        next_map.nodes[ret][i].is_leaf = 0;

        next_map.nodes[ret][i].ptr = PHYS_MAP_NODE_NIL;

    }

    return ret;

}
