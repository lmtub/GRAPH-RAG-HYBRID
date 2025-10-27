static void static_write(void *opaque, hwaddr offset, uint64_t value,

                         unsigned size)

{

#ifdef SPY

    printf("%s: value %08lx written at " PA_FMT "\n",

                    __FUNCTION__, value, offset);

#endif

}
