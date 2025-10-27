static void rtl8139_write_buffer(RTL8139State *s, const void *buf, int size)

{

    if (s->RxBufAddr + size > s->RxBufferSize)

    {

        int wrapped = MOD2(s->RxBufAddr + size, s->RxBufferSize);



        /* write packet data */

        if (wrapped && s->RxBufferSize < 65536 && !rtl8139_RxWrap(s))

        {

            DEBUG_PRINT((">>> RTL8139: rx packet wrapped in buffer at %d\n", size-wrapped));



            if (size > wrapped)

            {

                cpu_physical_memory_write( s->RxBuf + s->RxBufAddr,

                                           buf, size-wrapped );

            }



            /* reset buffer pointer */

            s->RxBufAddr = 0;



            cpu_physical_memory_write( s->RxBuf + s->RxBufAddr,

                                       buf + (size-wrapped), wrapped );



            s->RxBufAddr = wrapped;



            return;

        }

    }



    /* non-wrapping path or overwrapping enabled */

    cpu_physical_memory_write( s->RxBuf + s->RxBufAddr, buf, size );



    s->RxBufAddr += size;

}
