package com.attendance.config;

import org.hibernate.community.dialect.SQLiteDialect;
import org.hibernate.dialect.identity.IdentityColumnSupport;
import org.hibernate.dialect.identity.IdentityColumnSupportImpl;

public class CustomSQLiteDialect extends SQLiteDialect {

    @Override
    public IdentityColumnSupport getIdentityColumnSupport() {
        return new IdentityColumnSupportImpl() {
            @Override
            public boolean supportsIdentityColumns() {
                return true;
            }

            @Override
            public boolean hasDataTypeInIdentityColumn() {
                return false;
            }

            @Override
            public String getIdentityColumnString(int type) {
                return "integer";
            }

            @Override
            public String getIdentitySelectString(String table, String column, int type) {
                return "select last_insert_rowid()";
            }
        };
    }
}
