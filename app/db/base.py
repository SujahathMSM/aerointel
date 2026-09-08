from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, pool_pre_ping=True)

SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    """FastAPI dependency: one DB session per request."""
    async with SessionFactory() as session:
        yield session
