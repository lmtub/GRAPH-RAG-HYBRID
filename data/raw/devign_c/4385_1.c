static void control_to_network(RDMAControlHeader *control)

{

    control->type = htonl(control->type);

    control->len = htonl(control->len);

    control->repeat = htonl(control->repeat);

}
