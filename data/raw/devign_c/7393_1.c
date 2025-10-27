void os_mem_prealloc(int fd, char *area, size_t memory, Error **errp)

{

    int i;

    size_t pagesize = getpagesize();



    memory = (memory + pagesize - 1) & -pagesize;

    for (i = 0; i < memory / pagesize; i++) {

        memset(area + pagesize * i, 0, 1);

    }

}
