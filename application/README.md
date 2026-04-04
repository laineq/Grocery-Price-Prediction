# GroceryCast Application

This is the Next.js application for GroceryCast.

## Run locally

```bash
cd application
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Deploy with Docker on EC2

This app can run on the same EC2 instance as Airflow because it only needs
runtime access to S3 app-output files.

### 1. Install Docker on Amazon Linux 2023

Amazon Linux 2023 uses `dnf` for package management. AWS docs:
- https://docs.aws.amazon.com/linux/al2023/ug/package-management.html
- https://docs.aws.amazon.com/linux/al2023/ug/managing-repos-os-updates.html

On the EC2 host:

```bash
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker
docker --version
```

### 2. Copy the application folder to EC2

From your repo root on the server:

```bash
cd application
cp .env.production.example .env.production
```

Edit `.env.production` with your real values if needed.

### 3. Build the image

```bash
cd application
docker build -t grocerycast-application .
```

### 4. Run the container

```bash
docker run -d \
  --name grocerycast-application \
  --restart unless-stopped \
  -p 3000:3000 \
  --env-file .env.production \
  grocerycast-application
```

Then open:

- `http://<EC2_PUBLIC_IP>:3000`

### 5. Update after code changes

```bash
cd application
docker build -t grocerycast-application .
docker rm -f grocerycast-application
docker run -d \
  --name grocerycast-application \
  --restart unless-stopped \
  -p 3000:3000 \
  --env-file .env.production \
  grocerycast-application
```

### Notes

- This app reads S3 through the backend, so the EC2 instance needs AWS access.
- If the EC2 instance already has an IAM role with S3 permission, no AWS keys
  are needed inside `.env.production`.
- Airflow already uses port `8080`, so exposing the app on `3000` avoids port
  conflicts.
