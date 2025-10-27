static FlatView *address_space_get_flatview(AddressSpace *as)

{

    FlatView *view;



    rcu_read_lock();

    view = atomic_rcu_read(&as->current_map);

    flatview_ref(view);

    rcu_read_unlock();

    return view;

}
