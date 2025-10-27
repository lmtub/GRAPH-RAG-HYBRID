static int protocol_version(VncState *vs, char *version, size_t len)

{

    char local[13];

    int maj, min;



    memcpy(local, version, 12);

    local[12] = 0;



    if (sscanf(local, "RFB %03d.%03d\n", &maj, &min) != 2) {

	vnc_client_error(vs);

	return 0;

    }



    vnc_write_u32(vs, 1); /* None */

    vnc_flush(vs);



    vnc_read_when(vs, protocol_client_init, 1);



    return 0;

}
