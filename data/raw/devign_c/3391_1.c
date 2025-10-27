static void network_to_control(RDMAControlHeader *control)

{

    control->type = ntohl(control->type);

    control->len = ntohl(control->len);

    control->repeat = ntohl(control->repeat);

}
