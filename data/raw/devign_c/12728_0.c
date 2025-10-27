static void qemu_rdma_dump_id(const char *who, struct ibv_context *verbs)

{

    struct ibv_port_attr port;



    if (ibv_query_port(verbs, 1, &port)) {

        error_report("Failed to query port information");

        return;

    }



    printf("%s RDMA Device opened: kernel name %s "

           "uverbs device name %s, "

           "infiniband_verbs class device path %s, "

           "infiniband class device path %s, "

           "transport: (%d) %s\n",

                who,

                verbs->device->name,

                verbs->device->dev_name,

                verbs->device->dev_path,

                verbs->device->ibdev_path,

                port.link_layer,

                (port.link_layer == IBV_LINK_LAYER_INFINIBAND) ? "Infiniband" :

                 ((port.link_layer == IBV_LINK_LAYER_ETHERNET) 

                    ? "Ethernet" : "Unknown"));

}
