static void phys_page_set(AddressSpaceDispatch *d,

                          hwaddr index, hwaddr nb,

                          uint16_t leaf)

{

    /* Wildly overreserve - it doesn't matter much. */

    phys_map_node_reserve(3 * P_L2_LEVELS);



    phys_page_set_level(&d->phys_map, &index, &nb, leaf, P_L2_LEVELS - 1);

}
