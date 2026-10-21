import { text, sqliteTable } from "drizzle-orm/sqlite-core";

export const panelSnapshots = sqliteTable("panel_snapshots", {
  id: text("id").primaryKey(),
  payload: text("payload").notNull(),
  updatedAt: text("updated_at").notNull(),
});
