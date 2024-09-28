import { SpecialCardModel } from "@/types/cards";

type Props = {
  card: SpecialCardModel;
  count?: number;
};

export function SpecialCard({ card, count }: Props) {
  return (
    <div className="bg-white text-gray-900 rounded-lg p-3 border border-gray-200 w-[120px] h-[160px] flex flex-col justify-between relative">
      {/* Count badge in top right */}
      {count !== undefined && count > 1 && (
        <div className="absolute top-1 right-1 text-white text-[10px] font-bold rounded-full w-7 h-7 flex items-center justify-center border-2 border-white shadow-sm z-10 bg-blue-500">
          <span className="leading-none">{count}</span>
        </div>
      )}

      <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide py-1">Special</div>

      <div className="text-center flex-1 flex items-center justify-center">
        <div className="text-lg font-bold">
          {card.name === "JUST_SAY_NO" ? "Just Say No" : card.name?.replace(/_/g, " ") || "Special"}
        </div>
      </div>

      <div></div>
    </div>
  );
}
