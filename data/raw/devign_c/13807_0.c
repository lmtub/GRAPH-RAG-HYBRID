static void prom_set(uint32_t* prom_buf, int index, const char *string, ...)

{

    va_list ap;

    int32_t table_addr;



    if (index >= ENVP_NB_ENTRIES)

        return;



    if (string == NULL) {

        prom_buf[index] = 0;

        return;

    }



    table_addr = sizeof(int32_t) * ENVP_NB_ENTRIES + index * ENVP_ENTRY_SIZE;

    prom_buf[index] = tswap32(ENVP_ADDR + table_addr);



    va_start(ap, string);

    vsnprintf((char *)prom_buf + table_addr, ENVP_ENTRY_SIZE, string, ap);

    va_end(ap);

}
