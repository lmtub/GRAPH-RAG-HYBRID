static int qemu_rdma_resolve_host(RDMAContext *rdma, Error **errp)

{

    int ret;

    struct addrinfo *res;

    char port_str[16];

    struct rdma_cm_event *cm_event;

    char ip[40] = "unknown";

    struct addrinfo *e;



    if (rdma->host == NULL || !strcmp(rdma->host, "")) {

        ERROR(errp, "RDMA hostname has not been set");

        return -1;

    }



    /* create CM channel */

    rdma->channel = rdma_create_event_channel();

    if (!rdma->channel) {

        ERROR(errp, "could not create CM channel");

        return -1;

    }



    /* create CM id */

    ret = rdma_create_id(rdma->channel, &rdma->cm_id, NULL, RDMA_PS_TCP);

    if (ret) {

        ERROR(errp, "could not create channel id");

        goto err_resolve_create_id;

    }



    snprintf(port_str, 16, "%d", rdma->port);

    port_str[15] = '\0';



    ret = getaddrinfo(rdma->host, port_str, NULL, &res);

    if (ret < 0) {

        ERROR(errp, "could not getaddrinfo address %s", rdma->host);

        goto err_resolve_get_addr;

    }



    for (e = res; e != NULL; e = e->ai_next) {

        inet_ntop(e->ai_family,

            &((struct sockaddr_in *) e->ai_addr)->sin_addr, ip, sizeof ip);

        DPRINTF("Trying %s => %s\n", rdma->host, ip);



        /* resolve the first address */

        ret = rdma_resolve_addr(rdma->cm_id, NULL, e->ai_addr,

                RDMA_RESOLVE_TIMEOUT_MS);

        if (!ret) {

            goto route;

        }

    }



    ERROR(errp, "could not resolve address %s", rdma->host);

    goto err_resolve_get_addr;



route:

    qemu_rdma_dump_gid("source_resolve_addr", rdma->cm_id);



    ret = rdma_get_cm_event(rdma->channel, &cm_event);

    if (ret) {

        ERROR(errp, "could not perform event_addr_resolved");

        goto err_resolve_get_addr;

    }



    if (cm_event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {

        ERROR(errp, "result not equal to event_addr_resolved %s",

                rdma_event_str(cm_event->event));

        perror("rdma_resolve_addr");

        goto err_resolve_get_addr;

    }

    rdma_ack_cm_event(cm_event);



    /* resolve route */

    ret = rdma_resolve_route(rdma->cm_id, RDMA_RESOLVE_TIMEOUT_MS);

    if (ret) {

        ERROR(errp, "could not resolve rdma route");

        goto err_resolve_get_addr;

    }



    ret = rdma_get_cm_event(rdma->channel, &cm_event);

    if (ret) {

        ERROR(errp, "could not perform event_route_resolved");

        goto err_resolve_get_addr;

    }

    if (cm_event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {

        ERROR(errp, "result not equal to event_route_resolved: %s",

                        rdma_event_str(cm_event->event));

        rdma_ack_cm_event(cm_event);

        goto err_resolve_get_addr;

    }

    rdma_ack_cm_event(cm_event);

    rdma->verbs = rdma->cm_id->verbs;

    qemu_rdma_dump_id("source_resolve_host", rdma->cm_id->verbs);

    qemu_rdma_dump_gid("source_resolve_host", rdma->cm_id);

    return 0;



err_resolve_get_addr:

    rdma_destroy_id(rdma->cm_id);

    rdma->cm_id = NULL;

err_resolve_create_id:

    rdma_destroy_event_channel(rdma->channel);

    rdma->channel = NULL;



    return -1;

}
