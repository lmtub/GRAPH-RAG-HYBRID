bool qht_reset_size(struct qht *ht, size_t n_elems)

{

    struct qht_map *new;

    struct qht_map *map;

    size_t n_buckets;

    bool resize = false;



    n_buckets = qht_elems_to_buckets(n_elems);



    qemu_mutex_lock(&ht->lock);

    map = ht->map;

    if (n_buckets != map->n_buckets) {

        new = qht_map_create(n_buckets);

        resize = true;

    }



    qht_map_lock_buckets(map);

    qht_map_reset__all_locked(map);

    if (resize) {

        qht_do_resize(ht, new);

    }

    qht_map_unlock_buckets(map);

    qemu_mutex_unlock(&ht->lock);



    return resize;

}
