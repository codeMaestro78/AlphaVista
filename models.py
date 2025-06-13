from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences = relationship("UserPreference", back_populates="user")
    watchlists = relationship("Watchlist", back_populates="user")
    portfolios = relationship("Portfolio", back_populates="user")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    theme = Column(String, default="light")
    default_timeframe = Column(String, default="1mo")
    chart_type = Column(String, default="candlestick")
    indicators = Column(JSON)
    email_alerts = Column(Boolean, default=True)
    user = relationship("User", back_populates="preferences")

class Watchlist(Base):
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    symbols = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="watchlists")

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    holdings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="portfolios")

class HistoricalAnalysis(Base):
    __tablename__ = "historical_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String)
    analysis_type = Column(String)
    analysis_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    parameters = Column(JSON) 