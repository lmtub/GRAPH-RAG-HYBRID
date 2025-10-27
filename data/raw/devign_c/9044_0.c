static void destroy_all_mappings(void)

{

    destroy_l2_mapping(&phys_map, P_L2_LEVELS - 1);

    phys_map_nodes_reset();

}
