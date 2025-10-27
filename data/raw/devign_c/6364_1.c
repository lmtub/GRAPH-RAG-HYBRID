static void test_server_free(TestServer *server)

{

    int i;



    qemu_chr_delete(server->chr);



    for (i = 0; i < server->fds_num; i++) {

        close(server->fds[i]);

    }



    if (server->log_fd != -1) {

        close(server->log_fd);

    }



    unlink(server->socket_path);

    g_free(server->socket_path);





    g_free(server->chr_name);

    g_free(server);

}
