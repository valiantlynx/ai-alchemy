docker build -t ai-alchemy-image .
docker run --name ai-alchemy-container -d -p 8000:8000 -v $(pwd):/code ai-alchemy-image

#connect to turborepo
git subtree add --prefix=apps/ai-alchemy https://github.com/valiantlynx/ai-alchemy.git main --squash
git subtree pull --prefix=apps/ai-alchemy https://github.com/valiantlynx/ai-alchemy.git main --squash
git subtree push --prefix=apps/ai-alchemy https://github.com/valiantlynx/ai-alchemy.git main