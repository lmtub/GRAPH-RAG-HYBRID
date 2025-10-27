static void *legacy_s390_alloc(size_t size)

{

    void *mem;



    mem = mmap((void *) 0x800000000ULL, size,

               PROT_EXEC|PROT_READ|PROT_WRITE,

               MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED, -1, 0);

    return mem == MAP_FAILED ? NULL : mem;

}
