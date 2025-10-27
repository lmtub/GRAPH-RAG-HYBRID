void phys_mem_set_alloc(void *(*alloc)(size_t))

{

    phys_mem_alloc = alloc;

}
