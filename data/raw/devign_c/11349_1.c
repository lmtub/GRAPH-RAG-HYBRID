static void rx_stop_cont_test(const QVirtioBus *bus, QVirtioDevice *dev,

                              QGuestAllocator *alloc, QVirtQueue *vq,

                              int socket)

{

    uint64_t req_addr;

    uint32_t free_head;

    char test[] = "TEST";

    char buffer[64];

    int len = htonl(sizeof(test));

    struct iovec iov[] = {

        {

            .iov_base = &len,

            .iov_len = sizeof(len),

        }, {

            .iov_base = test,

            .iov_len = sizeof(test),

        },

    };

    int ret;



    req_addr = guest_alloc(alloc, 64);



    free_head = qvirtqueue_add(vq, req_addr, 64, true, false);

    qvirtqueue_kick(bus, dev, vq, free_head);



    qmp("{ 'execute' : 'stop'}");



    ret = iov_send(socket, iov, 2, 0, sizeof(len) + sizeof(test));

    g_assert_cmpint(ret, ==, sizeof(test) + sizeof(len));



    /* We could check the status, but this command is more importantly to

     * ensure the packet data gets queued in QEMU, before we do 'cont'.

     */

    qmp("{ 'execute' : 'query-status'}");

    qmp("{ 'execute' : 'cont'}");



    qvirtio_wait_queue_isr(bus, dev, vq, QVIRTIO_NET_TIMEOUT_US);

    memread(req_addr + VNET_HDR_SIZE, buffer, sizeof(test));

    g_assert_cmpstr(buffer, ==, "TEST");



    guest_free(alloc, req_addr);

}
