host_memory_backend_can_be_deleted(UserCreatable *uc, Error **errp)

{

    MemoryRegion *mr;



    mr = host_memory_backend_get_memory(MEMORY_BACKEND(uc), errp);

    if (memory_region_is_mapped(mr)) {

        return false;

    } else {

        return true;

    }

}
