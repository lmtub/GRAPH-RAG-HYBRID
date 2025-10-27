static void map_exec(void *addr, long size)

{

    DWORD old_protect;

    VirtualProtect(addr, size,

                   PAGE_EXECUTE_READWRITE, &old_protect);

    

}
