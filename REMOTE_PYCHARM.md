# PyCharm Remote Server Config

The following configuration was done using:
- PyCharm Professional Edition 2021.2.2
- Google Cloud `n1-highmem-2` VM running in zone `us-west1-b`
- macOS Big Sur version 11.5.1

1. Follow [Getting a Static IP Address][1] to reserve a static IP address
    - [Reserve a new static external IP address][2] is an alternate tutorial
2. Follow [SSH into Google Cloud][3] to:
    1. Generate a public key
    2. Add it to your Google Cloud VM
    3. Test `ssh`'ing in directly
3. Follow [Create a remote server configuration][4] to configure a remote server
    - Server name: \*your Google Cloud server hostname\*
    - Protocol: SFTP
    - SSH Configuration
        - Host: \*your static IP address\*
        - User name: \*your google cloud user.name\*
        - Port: 22
        - Local port: (leave as default of `<Dynamic>`)
        - Auth type: OpenSSH config and authentication agent
        - Connection Parameters: (leave as default)
    - Root path: \*your server's home directory\*
    - Uncheck "Visible only for this project"
        - Necessary for remote interpreter
    - Everything else: (leave as default)
4. Now set up [Mappings][5] ([other info here][6]) and a remote interpreter

Good luck to you!  These things always have hiccups.

[1]: https://github.com/cs231n/gcloud#getting-a-static-ip-address
[2]: https://cloud.google.com/compute/docs/ip-addresses/reserve-static-external-ip-address#reserve_new_static
[3]: https://www.siteyaar.com/google-cloud-ssh/#mac
[4]: https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html
[5]: https://www.jetbrains.com/help/pycharm/deployment-mappings-tab.html
[6]: https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html#mapping
