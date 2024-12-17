import datetime
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def generate_certificates(cert_dir="certs"):
    """Generate self-signed certificates with proper extensions for HTTPS proxy"""
    # Create certificates directory if it doesn't exist
    if not os.path.exists(cert_dir):
        print("Making: ", cert_dir)
        os.makedirs(cert_dir)

    # Generate private key
    ca_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,  # Increased key size for better security
    )

    # Generate public key
    ca_public_key = ca_private_key.public_key()

    # CA BEGIN
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "Proxy Pilot CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Proxy Pilot"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Development"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "UK"),
        ]
    )

    builder = x509.CertificateBuilder()
    builder = builder.subject_name(name)
    builder = builder.issuer_name(name)
    builder = builder.public_key(ca_public_key)
    builder = builder.serial_number(x509.random_serial_number())
    builder = builder.not_valid_before(datetime.datetime.utcnow())
    builder = builder.not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=3650)  # 10 years
    )

    builder = builder.add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=True,
    )

    builder = builder.add_extension(
        x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=True,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=True,  # This is a CA
            crl_sign=True,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=True,
    )

    ca_cert = builder.sign(
        private_key=ca_private_key,
        algorithm=hashes.SHA256(),
    )

    # Save CA certificate and key

    with open("certs/ca.crt", "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    with open("certs/ca.key", "wb") as f:
        f.write(
            ca_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    # CA END

    # SERVER BEGIN

    # Generate new certificate for domain
    server_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,  # 2048 bits is sufficient for domain certs
    )

    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "Proxy Pilot CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Proxy Pilot Generated"),
        ]
    )

    builder = x509.CertificateBuilder()
    builder = builder.subject_name(name)
    builder = builder.issuer_name(ca_cert.subject)
    builder = builder.public_key(server_key.public_key())
    builder = builder.serial_number(x509.random_serial_number())
    builder = builder.not_valid_before(datetime.datetime.utcnow())
    builder = builder.not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))

    # Add domain to SAN
    builder = builder.add_extension(
        x509.SubjectAlternativeName([x509.DNSName("localhost")]),
        critical=False,
    )

    # Add extended key usage
    builder = builder.add_extension(
        x509.ExtendedKeyUsage(
            [
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]
        ),
        critical=False,
    )

    # Basic constraints (not a CA)
    builder = builder.add_extension(
        x509.BasicConstraints(ca=False, path_length=None),
        critical=True,
    )

    certificate = builder.sign(
        private_key=ca_private_key,
        algorithm=hashes.SHA256(),
    )

    with open("certs/server.crt", "wb") as f:
        f.write(certificate.public_bytes(serialization.Encoding.PEM))

    with open("certs/server.key", "wb") as f:
        f.write(
            server_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    print("Certificates generated successfully in the 'certs' directory")
    print("\nTo trust these certificates:")
    print("\nOn macOS:")
    print(
        "security add-trusted-cert -r trustRoot "
        "-k ~/Library/Keychains/login.keychain certs/server.crt"
    )
    print("\nOn Windows (PowerShell):")
    print(
        'Import-Certificate -FilePath "certs\\server.crt" '
        "-CertStoreLocation Cert:\\CurrentUser\\Root"
    )
    print("\nOn Linux:")
    print("sudo cp certs/server.crt /usr/local/share/ca-certificates/proxy-pilot.crt")
    print("sudo update-ca-certificates")
    print("\nFor VS Code, add to settings.json:")
    print(
        """{
    "http.proxy": "https://localhost:8989",
    "http.proxySupport": "on",
    "github.copilot.advanced": {
        "debug.testOverrideProxyUrl": "https://localhost:8989",
        "debug.overrideProxyUrl": "https://localhost:8989"
    }
}"""
    )


if __name__ == "__main__":
    generate_certificates()
