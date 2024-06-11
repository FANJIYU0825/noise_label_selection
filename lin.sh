ldap_tls_reqcert = never
id_provider = ldap
auth_provider = ldap
chpass_provider = ldap
sudo_provider = ldap
#IP address of the NAS
ldap_uri = ldap://192.168.1.129
#Base DN from our LDAP server configuration.
ldap_search_base=dc=home,dc=lab
#Bind DN from our LDAP server configuration.
ldap_default_bind_dn = uid=root,cn=users,dc=home,dc=lab
ldap_default_authtok_type = dk ru8 xu/6
#Password from our LDAP server configuration.
ldap_default_authtok = dk ru8 xu/6
cache_credentials = True
use_fully_qualified_names = False
[sssd]
config_file_version = 2
services = nss,pam
#FQDN from LDAP server
domains = NTNUKDD
