void virtio_net_set_config_size(VirtIONet *n, uint32_t host_features)

{

    int i, config_size = 0;


    for (i = 0; feature_sizes[i].flags != 0; i++) {

        if (host_features & feature_sizes[i].flags) {

            config_size = MAX(feature_sizes[i].end, config_size);

        }

    }

    n->config_size = config_size;

}