int ff_socket_nonblock(int socket, int enable)

{

#ifdef __MINGW32__

   return ioctlsocket(socket, FIONBIO, &enable);

#else

   if (enable)

      return fcntl(socket, F_SETFL, fcntl(socket, F_GETFL) | O_NONBLOCK);

   else

      return fcntl(socket, F_SETFL, fcntl(socket, F_GETFL) & ~O_NONBLOCK);

#endif

}
