int ff_network_init(void)

{

#if HAVE_WINSOCK2_H

    WSADATA wsaData;

#endif



    if (!ff_network_inited_globally)

        av_log(NULL, AV_LOG_WARNING, "Using network protocols without global "

                                     "network initialization. Please use "

                                     "avformat_network_init(), this will "

                                     "become mandatory later.\n");

#if HAVE_WINSOCK2_H

    if (WSAStartup(MAKEWORD(1,1), &wsaData))

        return 0;

#endif

    return 1;

}
