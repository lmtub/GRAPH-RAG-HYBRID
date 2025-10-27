static void phys_page_set(AddressSpaceDispatch *d,

                          target_phys_addr_t index, target_phys_addr_t nb,

                          uint16_t leaf)

{

    /* Wildly overreserve - it doesn't matter much. */

    phys_map_node_reserve(3 * P_L2_LEVELS);



    phys_page_set_level(&d->phys_map, &index, &nb, leaf, P_L2_LEVELS - 1);

}
