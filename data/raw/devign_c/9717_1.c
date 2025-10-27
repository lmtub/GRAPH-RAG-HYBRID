static void address_space_update_topology(AddressSpace *as)

{

    FlatView old_view = as->current_map;

    FlatView new_view = generate_memory_topology(as->root);



    address_space_update_topology_pass(as, old_view, new_view, false);

    address_space_update_topology_pass(as, old_view, new_view, true);



    as->current_map = new_view;

    flatview_destroy(&old_view);

    address_space_update_ioeventfds(as);

}
