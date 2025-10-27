static void memory_region_update_container_subregions(MemoryRegion *subregion)

{

    hwaddr offset = subregion->addr;

    MemoryRegion *mr = subregion->container;

    MemoryRegion *other;



    memory_region_transaction_begin();



    memory_region_ref(subregion);

    QTAILQ_FOREACH(other, &mr->subregions, subregions_link) {

        if (subregion->may_overlap || other->may_overlap) {

            continue;

        }

        if (int128_ge(int128_make64(offset),

                      int128_add(int128_make64(other->addr), other->size))

            || int128_le(int128_add(int128_make64(offset), subregion->size),

                         int128_make64(other->addr))) {

            continue;

        }

#if 0

        printf("warning: subregion collision %llx/%llx (%s) "

               "vs %llx/%llx (%s)\n",

               (unsigned long long)offset,

               (unsigned long long)int128_get64(subregion->size),

               subregion->name,

               (unsigned long long)other->addr,

               (unsigned long long)int128_get64(other->size),

               other->name);

#endif

    }

    QTAILQ_FOREACH(other, &mr->subregions, subregions_link) {

        if (subregion->priority >= other->priority) {

            QTAILQ_INSERT_BEFORE(other, subregion, subregions_link);

            goto done;

        }

    }

    QTAILQ_INSERT_TAIL(&mr->subregions, subregion, subregions_link);

done:

    memory_region_update_pending |= mr->enabled && subregion->enabled;

    memory_region_transaction_commit();

}
