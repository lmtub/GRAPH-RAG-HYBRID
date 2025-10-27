static FlatView *address_space_get_flatview(AddressSpace *as)

{

    FlatView *view;



    qemu_mutex_lock(&flat_view_mutex);

    view = as->current_map;

    flatview_ref(view);

    qemu_mutex_unlock(&flat_view_mutex);

    return view;

}
