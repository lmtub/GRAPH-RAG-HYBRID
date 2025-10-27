static int chr_can_read(void *opaque)

{

    SCLPConsole *scon = opaque;



    return SIZE_BUFFER_VT220 - scon->iov_data_len;

}
