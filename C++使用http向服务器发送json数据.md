```cpp
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

int main()
{
		int sockfd = socket(AF_INET, SOCK_STREAM, 0);   // IPv4
		struct sockaddr_in server_addr;
		bzero(&server_addr, sizeof(server_addr));	
		server_addr.sin_family = AF_INET;
		const char *ip = "127.0.0.1";
		server_addr.sin_addr.s_addr = inet_addr(ip);	
		server_addr.sin_port = htons(1234);    //port
		int err_log = connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr));
		if (err_log != 0)
		{
			perror("connect");
			close(sockfd);
			return -1;
		}
		// message
		string pre = "POST /home/ubuntu/ HTTP/1.1\r\n"
			"Host: 127.0.0.1:1234\r\n"
			"Content-Type: application/json\r\n"
			"Content-Length: ";
		string json = "{\"item1\":\"val1\", \"item2\":6, \"image_nums\":[";
		ifstream ifs("image.txt");
		int tmp;
		while(ifs >> tmp) {
			if(ifs.eof()) break;
			string ss = to_string(tmp);
			json += ss + ",";
		}
		ifs.close();
		json = json.substr(0, json.length()-1) + "]" + "}";
		pre += to_string(json.length()) + "\r\n\r\n" + json;
		int send_ret = write(sockfd, pre.c_str(), pre.size());
		cout << "send " << send_ret << endl;
		// response
		char recv_buf[8 * 1024] = {0}; 
		recv(sockfd, recv_buf, sizeof(recv_buf), 0);
		FILE *fd = fopen("response.json", "wb+");
		fwrite(recv_buf, 1, bytes_received, fd);
		fclose(fd);
		close(sockfd);
		return 0;
}
```
