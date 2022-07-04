- 使用gethostbyname，返回值的结构如下：
> /\* Description of data base entry for a single host.  \*/
>struct hostent
{
  char *h_name;			/\* Official name of host.  \*/
  char **h_aliases;		/\* Alias list.  \*/
  int h_addrtype;		/\* Host address type.  \*/
  int h_length;			/\* Length of address.  \*/
  char **h_addr_list;		/\* List of addresses from name server.  \*/
#ifdef __USE_MISC
> \# define	h_addr	h_addr_list[0] /\* Address, for backward compatibility.\*/
>#endif
};
```cpp
#include <netdb.h>  // gethostbyname
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>

using namespace std;

int main()
{
		struct hostent* phost = gethostbyname("www.baidu.com");
		char* ip;
		if (NULL == phost)
	    {
	    	cout << "gethostbyname error : " <<  errno << " : " << strerror(errno) << endl;
	    	return -1;
	    }
		//inet_ntop(phost->h_addrtype,  phost->h_addr, ip, 17);
		for(int i=0; phost->h_addr_list[i]; i++) {
			ip = inet_ntoa( *(struct in_addr*)phost->h_addr_list[0] );
			cout << ip << endl;
		}
		return 0;
}
```
