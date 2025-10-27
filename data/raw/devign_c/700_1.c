static void free_device_list(AVOpenCLDeviceList *device_list)

{

    int i, j;

    if (!device_list)

        return;

    for (i = 0; i < device_list->platform_num; i++) {

        if (!device_list->platform_node[i])

            continue;

        for (j = 0; j < device_list->platform_node[i]->device_num; j++) {

            av_freep(&(device_list->platform_node[i]->device_node[j]->device_name));

            av_freep(&(device_list->platform_node[i]->device_node[j]));

        }

        av_freep(&device_list->platform_node[i]->device_node);

        av_freep(&(device_list->platform_node[i]->platform_name));

        av_freep(&device_list->platform_node[i]);

    }

    av_freep(&device_list->platform_node);

    device_list->platform_num = 0;

}
